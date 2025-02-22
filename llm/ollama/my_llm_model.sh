./bin/ollama serve &
pid=$!
sleep 5
echo "Pulling llava:7b-v1.5-fp16 model... "
until ollama pull llava:7b-v1.5-fp16; do
  echo "Retrying in 3 seconds..."
  sleep 10
done
wait $pid
