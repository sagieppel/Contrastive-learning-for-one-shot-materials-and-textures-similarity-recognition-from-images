while true
do
   echo "Restart training"
   python Train.py
   echo "Crushed"
done


# Run in loop to in case of crush (by defult the script should start from last saved point)
# You can use this if your program tend to crash once every few thousand steps (OOM) and you dont want to reduce batch size
