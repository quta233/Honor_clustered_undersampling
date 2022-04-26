import os
import torch
from torch import nn, optim
from torchvision.io import read_image
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import time
import numpy as np
from vgg import VGG16
from ZFNet import ZFNet
from data_county import Dataset_County
import utils_county
import sys
import math

# Define hyper-parameters
learning_rate = 2e-5
batch_size = 32
max_e = 1000
min_e = 100
Random_seed = 12345
add_epoch = 25

NETWORK_FOLDER = "/work/windmills/256/"
NETWORK_FOLDER2 = "/mnt/windmills/images/2017/IA/tif/256/"
SAVE_WORK = "/work/windmills/"

# Define training function
def train(dataloader, epoch):
    # Initialize
    start = time.time()
    size = len(dataloader.dataset)

    model.train()
    loss = 0

    for batch, (X, y, y_m) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # if y > 1:
        #     y = 1
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Plot to tenserboard
        writer.add_scalar("train/loss", loss, batch + len(dataloader) * epoch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            now = round(time.time() - start)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {now}")

    print(f"Epoch {epoch + 1} training done!\n")


# Define validation function
def validate(dataloader, epoch, multi):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0

    model.eval()
    confusion_m = np.array([[0, 0], [0, 0]])
    with torch.no_grad():
        for X, y, y_m in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #print(pred, pred.dtype, pred.argmax(1), y.dtype, y)
            valid_loss += loss_fn(pred, y).item()
            pred_label = pred.argmax(1)
            if multi:
                for i in range(len(pred_label)):
                    if pred_label[i] > 1:
                        pred_label[i] = 1
            correct += (pred_label == y).type(torch.float).sum().item()
            confusion_m += confusion_matrix(y.cpu(), pred_label.cpu(), labels=[0, 1])
    tp, fn, fp, tn = confusion_m.ravel()
    sensitivity = tp / (tp + fn)
    positive_acc = tp / (tp + fp)
    avg_loss = valid_loss / num_batches
    accuracy = correct / size
    print(f"Valid accuracy: {(100 * accuracy):>0.1f}%, Valid avg loss: {avg_loss:>8f}")
    print(f"Sensitivity: {(100 * sensitivity):>0.1f}%, Positive accuracy: {(100 * positive_acc):>0.1f}%")
    print(f"Epoch {epoch + 1} validation done!\n")

    # writer.add_scalars(f'valid', {
    #     "valid/accuracy": 100 * accuracy,
    #     "valid/sensitivity": 100 * sensitivity,
    #     "valid/positive_accuracy": 100 * positive_acc,
    # }, epoch)

    # Plot with tensorboard
    writer.add_scalar("valid/loss", avg_loss, epoch)
    writer.add_scalar("valid/accuracy", 100 * accuracy, epoch)
    writer.add_scalar("valid/sensitivity", 100 * sensitivity, epoch)
    writer.add_scalar("valid/positive_accuracy", 100 * positive_acc, epoch)
    return avg_loss


# Define testing function
def test(dataloader, multi):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    confusion_m = np.array([[0, 0], [0, 0]])
    with torch.no_grad():
        for X, y, y_m in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_label = pred.argmax(1)
            if multi:
                for i in range(len(pred_label)):
                    if pred_label[i] > 1:
                        pred_label[i] = 1
            correct += (pred_label == y).type(torch.float).sum().item()
            confusion_m += confusion_matrix(y.cpu(), pred_label.cpu(), labels=[0, 1])

    tp, fn, fp, tn = confusion_m.ravel()
    sensitivity = tp / (tp + fn)
    positive_acc = tp / (tp + fp)
    avg_loss = test_loss / num_batches
    accuracy = correct / size
    print(f"Test accuracy: {(100 * accuracy):>0.1f}%, Test avg loss: {avg_loss:>8f} \n")
    print(f"Test Sensitivity: {(100 * sensitivity):>0.1f}%, Test Positive accuracy: {(100 * positive_acc):>0.1f}%")
    return confusion_m, sensitivity


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # load in command line
    nn_module = sys.argv[1]
    Random_seed = int(sys.argv[2])
    train_s = str(sys.argv[3])
    match_method = str(sys.argv[4])
    # input: python classifier,py vgg/zfnet random_seed undersampling_method feature_match

    folder_common = train_s + "_lr_" + str(learning_rate) + "_bs_" + str(batch_size) + \
                  "_se_" + str(Random_seed) + "_fe_" + match_method + "_nn_" + nn_module



    ## split data
    test_file = open("test_name.txt", "r")
    test_name = test_file.read()
    test_name = test_name.split(",")
    test_name.pop(-1)

    tvmaj_file = open("tv_maj.txt", "r")
    tvmaj_name = tvmaj_file.read()
    tvmaj_name = tvmaj_name.split(",")
    tvmaj_name.pop(-1)

    tvmin_file = open("tv_min.txt", "r")
    tvmin_name = tvmin_file.read()
    tvmin_name = tvmin_name.split(",")
    tvmin_name.pop(-1)
    trainmaj_name, trainmin_name, validate_name = utils_county.splitData(tvmaj_name, tvmin_name, Random_seed)

    CWD = os.getcwd()
    os.chdir(NETWORK_FOLDER)
    # print(len(train_name), len(validate_name), len(test_name))
    new_train = []
    new_label = []
    num_class = 2
    multi_boo= False
    if train_s == "random":
        new_train = utils_county.randomtrain(trainmaj_name, trainmin_name, Random_seed)
    else:
        ## bow_voc
        hist, maj = utils_county.bow_voc(trainmaj_name, match_method)
        if train_s == "bin":
            new_train = utils_county.bin(maj, trainmin_name, hist)
        elif train_s == "lin":
            new_train = utils_county.lin(maj, trainmin_name, hist)
        elif train_s == "multi":
            new_train, new_label, num_class = utils_county.multi(maj, trainmin_name, hist)
            multi_boo = True
            torch.backends.cudnn.benchmark = True
        else:
            print("no valid method")
            exit(0)

    print("num_class", num_class)
    if nn_module == "vgg":
        model = VGG16(num_classes=num_class).to(device)
    else:
        model = ZFNet(num_classes=num_class).to(device)

    model.to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # print(len(new_train), train_s)
    transform = transforms.Compose([transforms.ToTensor()])

    #print(Random_seed)
    if multi_boo:
        #print(len(new_label), len(new_train), new_label[:20])
        new_label = torch.from_numpy(new_label)
        train_data = Dataset_County(new_train, NETWORK_FOLDER, transform=transform, multi=True, multi_label=new_label)
    else:
        train_data = Dataset_County(new_train, NETWORK_FOLDER, transform=transform)

    validate_data = Dataset_County(validate_name, NETWORK_FOLDER, transform=transform)
    test_data = Dataset_County(test_name, NETWORK_FOLDER, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # os.chdir(SAVE_WORK)
    os.chdir(CWD)
    image_Folder = folder_common
    # Plot with tensorboard
    writer = SummaryWriter(comment=image_Folder)

    modelFolder = "./module/" + folder_common + "module.pth"
    # Train and test

    best_vloss = math.inf
    epochs = 0
    epo_best = 0
    if multi_boo:
        min_e = 150
        add_epoch = 50

    while (epochs < min_e or epochs < add_epoch + epo_best) and epochs < max_e:
        print("sampling: ", train_s, "seed: ", Random_seed, "method: ", match_method, "nn: ", nn_module)
        print(f"Epoch {epochs + 1}\n-------------------------------")
        torch.cuda.empty_cache()
        train(train_dataloader, epochs)
        cur_loss = validate(validate_dataloader, epochs, multi_boo)
        if cur_loss < best_vloss:
            best_vloss = cur_loss
            epo_best = epochs
        # Save model
        torch.save(model.state_dict(), modelFolder)
        print("Saved PyTorch Model State to model.pth")
        epochs += 1
    print("Training and validation done! Testing start ------------------")
    c_m, sensitivity = test(test_dataloader, multi_boo)
    file_name = "./confusion/" + folder_common + ".txt"
    np.savetxt(file_name, c_m)
    s_file = str(train_s) + "_s.txt"
    file = open(s_file, "a+")
    s = str(sensitivity)
    file.write(s)
    file.write("\n")
    file.close()
    print("Done!")

    # Save model
    torch.save(model.state_dict(), modelFolder)
    print("Saved PyTorch Model State to model.pth")

    # Close writer
    writer.flush()
    writer.close()
