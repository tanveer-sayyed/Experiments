./bin/ollama serve &
pid=$!
sleep 5
echo "Pulling mistral model... "
until ollama pull mistral; do
  echo "Retrying in 3 seconds..."
  sleep 10
done
wait $pid
