#!/usr/bin/env bash
# ¿Puede Ollama correr en esta máquina, y correr sobre la GPU?
#
# Mismo formato de puertas que scripts/try_vllm_rocm.sh, y por la misma razón: el
# resultado puede ser "no", y un "no" con el mensaje exacto de la puerta que falla vale
# mucho más en un white paper que "no funcionó".
#
# Con Ollama hay una trampa que no existe con vLLM: **Ollama no falla cuando no
# encuentra la GPU, se cae a CPU y sirve igual**. Un benchmark contra un Ollama en CPU
# no mide Ollama, mide el fallback, y el número saldría espectacularmente a favor de
# fox sin que nadie se dé cuenta hasta que un lector lo reproduzca. Por eso la puerta 2
# no pregunta "¿carga el modelo?" sino "¿en qué procesador quedó residente?", leído de
# `ollama ps`, que es el único sitio donde Ollama lo dice sin ambigüedad.
#
# La máquina es una Radeon 890M — gfx1150, RDNA 3.5, integrada (gfx_target_version
# 110500 en /sys/class/kfd/kfd/topology/nodes/1/properties). Las builds ROCm de Ollama
# empaquetan kernels para un conjunto fijo de targets; si gfx1150 no está,
# HSA_OVERRIDE_GFX_VERSION lo presenta como una RDNA3 discreta. Se prueba sin override
# y con él, en ese orden, porque si funciona sin override eso es lo que hay que
# publicar en la configuración documentada.
#
#   scripts/try_ollama_rocm.sh
#
# El modelo es el mismo GGUF Q8_0 de las tiradas ya publicadas de fox vs llama-server,
# importado con un Modelfile en vez de `ollama pull`. Dos motivos: no descarga nada, y
# deja el eje 1 (config-matched) exacto — mismo fichero, misma cuantización, mismo
# vocabulario que los otros tres motores. Un `ollama pull llama3.2:1b` traería Q4_K_M y
# la comparación ya no sería del serving layer.
set -uo pipefail

IMAGE="${IMAGE:-ollama/ollama:rocm}"
GGUF="${GGUF:-$HOME/.cache/ferrumox/models/llama-3.2-1b-instruct-q8_0.gguf}"
TAG="${TAG:-foxbench-l32-1b-q8}"
PORT="${PORT:-11439}"
CONT="${CONT:-fox-ollama-gate}"
# Ollama descarta las GPU integradas por defecto y se cae a CPU sin fallar; su propio
# log nombra el flag ("dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1").
# Aquí es 1 por defecto porque esta máquina no tiene otra GPU: sin él, el banco mediría
# Ollama en CPU contra fox en Vulkan, que no es una comparación, es un accidente.
# IGPU=0 reproduce el comportamiento por defecto, que también es un dato publicable.
IGPU="${IGPU:-1}"
# Directorio de datos propio: `ollama create` copia el GGUF a su almacén de blobs, así
# que esto duplica ~1.3 GB. Va a un sitio desechable y se borra al final — limpiar
# después de CADA prueba, no al final de la sesión.
DATA="${DATA:-$(mktemp -d -t ollama-gate-XXXX)}"
URL="http://127.0.0.1:$PORT"

[ -f "$GGUF" ] || { echo "no existe el GGUF: $GGUF"; exit 1; }

cleanup() {
  docker rm -f "$CONT" >/dev/null 2>&1
  # El almacén de blobs queda con permisos de root (el contenedor corre como root).
  docker run --rm -v "$DATA:/d" alpine sh -c 'rm -rf /d/..?* /d/.[!.]* /d/*' >/dev/null 2>&1
  rmdir "$DATA" 2>/dev/null
}
trap cleanup EXIT

echo "imagen  $IMAGE"
echo "modelo  $GGUF"
echo "datos   $DATA (se borra al salir)"
echo

# ---------------------------------------------------------------- puerta 1: descubrimiento
# Ollama decide GPU vs CPU una sola vez, al arrancar, y lo escribe en su log. Se lee de
# ahí y no de `rocminfo`: que ROCm vea el dispositivo no implica que Ollama lo acepte;
# tiene su propia lista de targets soportados y su propio umbral de VRAM.
start_server() {
  local override="$1"
  docker rm -f "$CONT" >/dev/null 2>&1
  local envs=()
  [ -n "$override" ] && envs=(-e "HSA_OVERRIDE_GFX_VERSION=$override")
  docker run -d --name "$CONT" \
    --device=/dev/dri --device=/dev/kfd --group-add video \
    -v "$DATA:/root/.ollama" -v "$GGUF:/models/model.gguf:ro" \
    -p "127.0.0.1:$PORT:11434" \
    "${envs[@]}" -e OLLAMA_DEBUG=1 -e OLLAMA_IGPU_ENABLE="$IGPU" \
    "$IMAGE" >/dev/null 2>&1 || return 1
  for _ in $(seq 1 60); do
    curl -sf -m 2 "$URL/api/version" >/dev/null 2>&1 && return 0
    docker ps -q -f "name=$CONT" | grep -q . || return 1
    sleep 1
  done
  return 1
}

gpu_verdict() {
  # Las líneas que importan, en orden de claridad decreciente:
  #   "inference compute ... library=ROCm"   -> aceptó la GPU y va a usarla
  #   "dropping integrated GPU"              -> la vio, la reconoció, y la descartó
  #   "amdgpu is not supported" / "no compatible GPUs" -> la rechazó, y dice por qué
  docker logs "$CONT" 2>&1 | grep -Ei \
    'inference compute|dropping|amdgpu|rocm|gfx|no compatible|unsupported|looking for compatible' \
    | grep -v 'server config' | tail -12
}

# El veredicto se lee SÓLO de la línea "inference compute", que es la que dice qué va a
# usar de verdad. La primera versión de esta comprobación casaba con cualquier línea que
# mencionara ROCm y dio un falso positivo: el log decía "dropping integrated GPU ...
# library=ROCm compute=gfx1150", o sea justo lo contrario de lo que se estaba
# afirmando. Que el nombre del backend aparezca en el log no significa que se use.
accepted_gpu() {
  docker logs "$CONT" 2>&1 | grep -i 'inference compute' | grep -qiv 'library=cpu'
}

CHOSEN=""
for ov in "" "11.0.0" "11.0.2"; do
  label="${ov:-sin override}"
  echo "=== puerta 1: ¿Ollama descubre la GPU?  [$label, OLLAMA_IGPU_ENABLE=${IGPU:-vacío}] ==="
  if ! start_server "$ov"; then
    echo "  el servidor no arrancó"
    docker logs "$CONT" 2>&1 | tail -8
    continue
  fi
  gpu_verdict | sed 's/^/  /'
  if accepted_gpu; then
    echo "  --> GPU aceptada con [$label]"
    CHOSEN="$ov"
    break
  fi
  echo "  --> se queda en CPU con [$label]"
  echo
done

if [ -z "$CHOSEN" ] && ! docker ps -q -f "name=$CONT" | grep -q .; then
  echo
  echo "Ollama no llegó a servir en ninguna configuración. Ese es el resultado a anotar."
  exit 1
fi

echo
echo "=== puerta 2: ¿el modelo queda residente en GPU o en CPU? ==="
# Sin esta puerta el resto del banco no vale nada: Ollama sirve igual en CPU y la
# comparación quedaría midiendo el fallback.
docker exec "$CONT" sh -c 'printf "FROM /models/model.gguf\n" > /root/Modelfile' || exit 1
if ! docker exec "$CONT" ollama create "$TAG" -f /root/Modelfile 2>&1 | tail -3; then
  echo "  falló la importación del GGUF"
  exit 1
fi

# `ollama ps` sólo lista modelos cargados, así que hay que provocar una carga antes.
curl -sf -m 300 "$URL/api/generate" -d "{\"model\":\"$TAG\",\"prompt\":\"hola\",\"stream\":false,\"options\":{\"num_predict\":8}}" \
  | head -c 300 | sed 's/^/  respuesta: /'
echo
echo
docker exec "$CONT" ollama ps 2>&1 | sed 's/^/  /'
# La columna PROCESSOR se localiza por su forma ("100% CPU" / "100% GPU"), no por
# posición: la última columna de `ollama ps` es "4 minutes from now" y contarla desde el
# final devuelve "from now", que es lo que informó la primera versión de este script.
PROC="$(docker exec "$CONT" ollama ps 2>/dev/null | grep -oE '[0-9]+%/?[0-9]*%? *(CPU|GPU)(/GPU)?' | head -1)"
echo "  --> procesador: ${PROC:-desconocido}"

echo
echo "=== puerta 3: ¿sirve la API compatible con OpenAI que usa el banco? ==="
# bench_burst.py habla /v1/chat/completions en streaming y lee usage.prompt_tokens y
# usage.prompt_tokens_details.cached_tokens. Lo segundo Ollama no lo expone; hay que
# saberlo ANTES de medir, porque un cached_tokens=0 en la tabla se lee como "no reusa
# prefijos" cuando en realidad significa "no lo reporta". El TTFT sí es comparable.
curl -sN -m 120 "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$TAG\",\"messages\":[{\"role\":\"user\",\"content\":\"di hola\"}],\"max_tokens\":16,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  2>&1 | tail -4 | sed 's/^/  /'

echo
echo "=== resumen ==="
echo "  override necesario : ${CHOSEN:-ninguno}"
echo "  procesador         : ${PROC:-desconocido}"
echo "  (anota ambos en docs/design/benchmark-plan-2026-08.md)"
