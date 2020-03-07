import matplotlib.pyplot as plt
import numpy as np
import sys, os, pickle
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from datetime import datetime

class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        # define the layers of the network -- place any
        #        network you want here
        pairs = zip(conf[:-1], conf[1:])
        self.fcs = nn.ModuleList(nn.Linear(this_layer, next_layer)
                                 for (this_layer, next_layer) in pairs)
  
    def forward(self, x):
        # define the activation filters that connect layers
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
        x = self.fcs[-1](x)
        return x

# grab arguments
if (len(sys.argv) > 1):
  new_sim = str(sys.argv[1])
  epoch_pow = np.int(sys.argv[1])
else:
  new_sim = "y"
  epoch_pow = 11
if (len(sys.argv) > 2):
  window_ind = np.int(sys.argv[2])
else:
  window_ind = 11
if (len(sys.argv) > 3):
  data_out = str(sys.argv[3])
else:
  data_out = "n"
if (len(sys.argv) > 4):
   str(sys.argv[4])
else:
  train = "y"

# setup directories
ddir = "/home/ML_Literacy"
os.makedirs(ddir, exist_ok=True)

# setup dictionary to store data for output later
odict = dict()

fi = ddir+"/data-train.csv"

din = pd.read_csv(fi) #input data
intensity = np.array(din["Intensity"])
labels = np.array(din["Label"])
timeky = np.array(din["Time_Yr"])

n_y = 1/(timeky[1] - timeky[0]) #frequency step

HUGE = 10000000
beta = 1000

n_window_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000])

n_window = np.int(n_window_list[window_ind])
print(n_window)

file_suffix = "net"

# step one -- build a training set from the data
n_samp = 10000 

n_k = int(n_y*n_window)

# list all indexes above window length
ind_all = np.arange(n_k,len(intensity)) #make sure window never runs off the data
ind_no2 = ind_all[labels[n_k:len(intensity)] != 2] #omit segments with event
                                                   #(this choice is problem-
                                                   #dependent)
ind_sam1 = np.random.choice(ind_no2, n_samp) #make scrambled versions of
                                             #ind_no2... n_samp of them
ind_sam = ind_sam1

dss = np.zeros([n_samp, n_k]) #initialize data arrays
dls = np.zeros(n_samp)

for (i, j) in enumerate(ind_sam): #dump data into those arrays
  dss[i] = intensity[j-n_k+1:j+1]
  dls[i] = labels[j]


def main(): #this is in main() for making the code platform-independent
  conf = [n_k, 80, 75, 25, 10, 2] #list of layer sizes
  net = Net(conf)
  #"Report card" for the network:
  output_fil = nn.CrossEntropyLoss()  # error function

  # step 2: pick an optimizer -- pytorch has many
  #This tells how to adjust the network weights to improve its performance:
  weight_opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,
                         weight_decay=0.00001)
  
  fair = 0.5
  fanr = 0.5

  n_epochs = np.power(2,epoch_pow)
  closses = np.zeros(n_epochs)
  closs_change = 1.0

  # if we have trained network already, keep it
  end_total = 0
  if (train == "y"):
    # step 3: train the model
    for epoch in range(n_epochs):
        start = datetime.now()
        closs = 0.0
        print("Starting epoch: {}/{}".format(epoch, n_epochs))
        perm = np.random.permutation(len(dss))

        chunk_size = n_samp
        n_chunks = int(len(dss)/chunk_size)
        first = True
        losso = torch.tensor(0.0)
        for i in np.arange(0, n_chunks):
            print("starting iteration: {}/{}, alpha(i, n): {}, {}".format(i+1, n_chunks, fair, fanr))
            #Convert training set into torch tensors/torch-friendly stuff
            ip, il = torch.tensor(dss[i:i+chunk_size]).float(), Variable(torch.tensor(dls[i:i+chunk_size]).long())
            weight_opt.zero_grad() #initialize optimizer
            out = net(ip) ###Feed data into network, get output
            ilyes = il[np.where(il==1)]
            ilno  = il[np.where(il==0)]
            outyes= out[np.where(il==1)]
            outno = out[np.where(il==0)]
            losso += output_fil(outyes, ilyes) + output_fil(outno,ilno) # ute the loss, collect it
        loss =  losso 
        loss.backward() ###Compute gradient w/ respect to all weights.
                        ###This will enable an optimization step.
                        ###Torch knows where the gradient is, so we didn't
                        ###need to write blah = loss.backward().
        weight_opt.step() ###Modify weights a little bit based on loss.backward
        closs += loss.item()
        end_seconds = (datetime.now()-start).seconds
        end_total+= end_seconds
        print("Epoch: {}, Duration:{}, Closs: {}".format(epoch, end_seconds, closs))
        print("Approximate time remaining: {}".format(((end_total)/(epoch+1))*(n_epochs-epoch)/3600))
        closses[epoch] = closs
        if (epoch > 10):
          closs_change = (closses[epoch-1] - closs)
          print("Loss Improvement: {}".format(closs_change))

    print("Training Complete")
    print("Dumping training weights to disk")
    weights_dict = {}
    for param in list(net.named_parameters()):
        print("Serializing Param", param[0])
        weights_dict[param[0]] = param[1]

    fo = ddir + "/net_train-weights"+file_suffix+".npy"
    print("Dumping training weights to disk: {}".format(fo))
    np.save(fo, weights_dict)

  else:    
    wweights_dict = {}
    wfile = ddir + "/Config-2-weights-mb19"+file_suffix+".npy"
    assert os.path.isfile(wfile), "Error: Invalid {} ".format(wfile)
    wweights_dict = np.load(wfile).item()      
    for param in net.named_parameters():
        if param[0] in wweights_dict.keys():
            print("Copying: ", param[0])
            param[1].data = wweights_dict[param[0]].data 
    print("Weights Loaded!")

  print("plotting losses")
  fig2 = plt.figure()
  plt.plot(np.arange(0,n_epochs), closses, color="blue")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  fig2.set_size_inches(18, 9)
  fo = ddir+"/net_train-loss"+file_suffix+".png"
  fig2.savefig(fo, dpi=100)

  # dump closses
  fo = ddir+"/net_train-loss"+file_suffix+".npy"
  np.save(fo, closses)
  odict["closses"] = closses

  # step 4: TEST the model to see if it's any good
  print("beginning test")
  fi = ddir+"/data-test.csv"

  tin = pd.read_csv(fi)
  intensity_t = np.array(tin["Intensity"])
  labels_t = np.array(tin["Label"])
  timeky_t = np.array(tin["Time_Yr"])

  ind_all = np.arange(n_k,len(intensity_t))
  ind_no2 = ind_all[labels_t[n_k:len(intensity_t)] != 2]
  ind_sam = ind_no2

  dss_t = np.zeros([len(ind_sam), n_k])
  dls_t = np.zeros(len(ind_sam))

  for (i, j) in enumerate(ind_sam):
    dss_t[i] = intensity_t[j-n_k+1:j+1]
    dls_t[i] = labels_t[j]


  dlt_t = np.zeros([len(labels_t)])
  correct = 0.0
  correct_i = 0.0
  correct_n = 0.0
  total = 0.0
  total_i = 0.0
  total_n = 0.0
  for i, dss_tt in enumerate(dss_t):
    if (i % 1000 == 0):
      print("starting iteration: {}/{}".format(i+1, len(dss_t)))
    ip, il = dss_tt, dls_t[i]
    ipo = torch.tensor(ip).float()
    outputs = net(Variable(ipo))
    _, predicted = torch.max(outputs.data, 0)
    dlt_t[ind_sam[i]] = predicted
    total += 1
    if (il == 0):
      total_n += 1
    if (il == 1):
      total_i += 1

    if (predicted == il):
      correct += 1
      if (il == 0):
        correct_n += 1
      if (il == 1):
        correct_i += 1
  
  print("test Accuracy: {}/{} ({}%), {} errors".format(correct, total, 100 * correct / total, total-correct))
  print("Inversion Accuracy: {}/{} ({}%), {} errors".format(correct_i, total_i, 100 * correct_i / total_i, total_i - correct_i))
  print("Nothing   Accuracy: {}/{} ({}%), {} errors".format(correct_n, total_n, 100 * correct_n / total_n, total_n - correct_n))

  odict["training_accuracy"] = np.array([correct, total, 100 * correct / total, total-correct])
  odict["training_accuracy_i"] = np.array([correct_i, total_i, 100 * correct_i / total_i, total_i - correct_i])
  odict["training_accuracy_n"] = np.array([correct_n, total_n, 100 * correct_n / total_n, total_n - correct_n])

  fo = ddir+"/predictions"+file_suffix+".npy"
  np.save(fo, dlt_t)

  tin["nn_prediction"] = dlt_t
  fo = ddir+"/simulation_workshop_test_prediction.csv"
  tin.to_csv(fo, sep=",", index=False)

  # now save these and plot them
  # plot intensity versus time
  fig3 = plt.figure()
  plt.plot(timeky_t, intensity_t, color="green")
  plt.step(timeky_t, labels_t, color="black")
  plt.step(timeky_t, dlt_t*0.6, color="blue")    

  plt.xlabel("Time (Yr)")
  plt.ylabel("Intensity and Labels")
  fig3.set_size_inches(27, 9)
  fo = ddir+"/label_plot-test"+file_suffix+".png"
  fig3.savefig(fo, dpi=100)
  #plt.show()

  
if __name__ == '__main__':
  main()
  print("done")
