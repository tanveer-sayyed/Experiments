./bin/ollama serve &
pid=$!
sleep 5
echo "Pulling mistral:7b-instruct-v0.3-fp16 model... "
until ollama pull mistral:7b-instruct-v0.3-fp16; do
  echo "Retrying in 3 seconds..."
  sleep 10
done
wait $pid