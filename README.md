# Federated Learning training framework for raspberriPi
A simple application that uses docker and gRPC to demonstrate fedrated learning


Few commands to fetech training and accumulates the data

Delete the previous folder, ** Double-check file name **

```
# rm -r ~/Desktop/Training/ANN/Client1/data
# scp -r pi@192.168.0.123:~/Desktop/FL-Client/data ~/Desktop/Training/ANN/Client1

# rm -r ~/Desktop/Training/ANN/Server/Models
# cp -r ~/Desktop/FL-Server/interface/Models ~/Desktop/Training/ANN/Server
```

