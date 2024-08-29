#%%
from cosfire_workflow_utils_12082024 import *
import json
from tqdm import tqdm
num = 1
# margin = 36
# alpha = 0.001
# epochs = 100
# batch_size = 32
# learning_rate = 0.01
# bitsize = 36
input_size = 372

data_path = f"./descriptor_set_{num}_train_valid_test.mat" # Path to the Train_valid_test.mat file
data_path_valid = f"descriptor_set_{num}_train_valid.mat" # Path to the Train_valid.mat file
data_path_test = f"descriptor_set_{num}_train_test.mat" # Path to the Train_test.mat file

output_dir = f'output{num}' 

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print(f"The directory {output_dir} already exists.")

print(output_dir)

#model 3 layers
class CosfireNet(nn.Module):
    def __init__(self, input_size, bitsize, l1_reg, l2_reg):
        super(CosfireNet, self).__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.hd = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.BatchNorm1d(300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.Tanh(),
            nn.Linear(200, bitsize),
            nn.Tanh()
        )


    def forward(self, x):        
        regularization_loss = 0.0
        for param in self.hd.parameters():
            regularization_loss += torch.sum(torch.abs(param)) * self.l1_reg  # L1 regularization
            regularization_loss += torch.sum(param ** 2) * self.l2_reg  # L2 regularization
        return self.hd(x), regularization_loss


# Data
class CosfireDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(preprocessing.normalize(dataframe.iloc[:, :-1].values), dtype=torch.float32)
        self.labels = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



# loss function
def DSHLoss(u, y, alpha, margin):
    #alpha = 1e-5  # m > 0 is a margin. The margin defines a radius around X1,X2. Dissimilar pairs contribute to the loss
    # function only if their distance is within this radius
    m = 2 * margin  # Initialize U and Y with the current batch's embeddings and labels
    y = y.int()
    # Create a duplicate y_label
    y = y.unsqueeze(1).expand(len(y),len(y))
    y_label = torch.ones_like(torch.empty(len(y), len(y)))
    y_label[y == y.t()] = 0
    dist = torch.cdist(u, u, p=2).pow(2)
    loss1 = 0.5 * (1 - y_label) * dist
    loss2 = 0.5 * y_label * torch.clamp(m - dist, min=0)
    B1 = torch.norm(torch.abs(u) - 1, p=1, dim=1)
    B2 = B1.unsqueeze(1).expand(len(y), len(y))
    loss3 = (B2 + B2.t()) * alpha
    minibatch_loss = torch.mean(loss1 + loss2 + loss3)
    return minibatch_loss


dic_labels = { 'Bent':2,
  'Compact':3,
    'FRI':0,
    'FRII':1
}

dic_labels_rev = { 2:'Bent',
                3:'Compact',
                  0:'FRI',
                  1: 'FRII'
              }

#input_size	   output_size	learning_rate	batch_size	alpha	  margin	l1_reg	       l2_reg
#372	       48	        0.01	        32	        0.00001	  48	    0.000000e+00	1.000000e-08	90.22


epochs = 5
learning_rate = 0.01
alpha = 0.00001
margin = 36
batch_size = 48
bitsize = 72
l1_reg = 1e-8
l2_reg = 1e-08

#List of Grid search parameters to iterate over

# learning_rate_values = [ 0.1, 0.01]
# alphas = [1e-03,  1e-5]
# margin_values = [24, 36, 48] 
# batch_size_values = [32, 48, 64]
# bitsize_values = [40,56]#[8, 16, 24, 32, 48]
# l2_reg_values = [0, 1e-08]
# l1_reg_values = [0, 1e-08]
                                                           
# Train Valid & Test data
train_df,valid_df,test_df = get_and_check_data_prev(data_path,data_path_valid,data_path_test,dic_labels)                                

# DataLoader for training set
train_dataset = CosfireDataset(train_df)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# DataLoader for validation set
valid_dataset = CosfireDataset(valid_df)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
                  
model = CosfireNet(input_size, bitsize, l1_reg, l2_reg)

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

# Lists to store training and validation losses
train_losses = []
train_losses_eval = []

val_losses = []
# Variables to keep track of the best model

# Train the loop
for _ in tqdm(range(epochs), desc='Training Progress', leave=True):
      model.train()
      total_train_loss = 0.0
      for _, (train_inputs, labels) in enumerate(train_dataloader):
         optimizer.zero_grad()
         train_outputs, reg_loss = model(train_inputs)
         loss = DSHLoss(u = train_outputs, y=labels, alpha = alpha, margin = margin) + reg_loss
         loss.backward()
         optimizer.step()
         total_train_loss += loss.item() * train_inputs.size(0)
      scheduler.step()

      # Calculate average training loss
      average_train_loss = total_train_loss / len(train_dataloader)
      train_losses.append(average_train_loss)

      # Validation loop
      model.eval()
      total_val_loss = 0.0
      with torch.no_grad():
         for val_inputs, val_labels in valid_dataloader:
            val_outputs, reg_loss = model(val_inputs)
            val_loss = DSHLoss(u = val_outputs, y=val_labels, alpha = alpha, margin = margin) + reg_loss
            total_val_loss += val_loss.item() * val_inputs.size(0)

      # Calculate average validation loss
      average_val_loss = total_val_loss / len(valid_dataloader)
      val_losses.append(average_val_loss)




plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.savefig(output_dir + '/Train_valid_curves.png')
plt.close()

#########################################################################
#################     Evaluate                                                  
##########################################################################

#model.eval()
valid_dataset_eval = CosfireDataset(valid_df)
valid_dataloader_eval = DataLoader(valid_dataset_eval, batch_size=batch_size, shuffle=False)

#valid_dataloader_eval = torch.utils.data.DataLoader(valid_dataset_eval,      sampler=ImbalancedDatasetSampler(valid_dataset_eval),batch_size=batch_size)

train_dataset_eval = CosfireDataset(train_df)
train_dataloader_eval = DataLoader(train_dataset_eval, batch_size=batch_size, shuffle=False)
#train_dataloader_eval = torch.utils.data.DataLoader(train_dataset_eval,      sampler=ImbalancedDatasetSampler(train_dataset_eval),batch_size=batch_size)

test_dataset = CosfireDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Lists to store predictions
predictions = []

# Perform predictions on the train set
with torch.no_grad():
      for train_inputs, _ in tqdm(train_dataloader_eval, desc='Predicting', leave=True):
         train_outputs,_ = model(train_inputs)
         predictions.append(train_outputs.numpy())

# Flatten the predictions
flat_predictions_train = [item for sublist in predictions for item in sublist]

# Append predictions to the df_train DataFrame
train_df['predictions'] = flat_predictions_train
train_df['label_name'] = train_df['label_code'].map(dic_labels_rev)
train_df.to_csv(output_dir +'/train_df.csv',index = False)

#################################################################

predictions = []
# Perform predictions on the valid set
with torch.no_grad():
      for valid_inputs, _ in tqdm(valid_dataloader_eval, desc='Predicting', leave=True):
         valid_outputs,_ = model(valid_inputs)
         predictions.append(valid_outputs.numpy())
# Flatten the predictions
flat_predictions_valid = [item for sublist in predictions for item in sublist]
# Append predictions to the valid_df DataFrame
valid_df['predictions'] = flat_predictions_valid
valid_df['label_name'] = valid_df['label_code'].map(dic_labels_rev)
valid_df.to_csv(output_dir +'/valid_df.csv',index = False)


#########################################################################
##Testing
#########################################################################
#Perform predictions on the testing set

predictions_test = []
with torch.no_grad():
      for test_inputs, _ in tqdm(test_dataloader, desc='Predicting', leave=True):
         test_outputs,_ = model(test_inputs)
         predictions_test.append(test_outputs.numpy())

# Flatten the predictions
flat_predictions_test = [item for sublist in predictions_test for item in sublist]

# Append predictions to the test_df DataFrame
test_df['predictions'] = flat_predictions_test
test_df['label_name'] = test_df['label_code'].map(dic_labels_rev)
test_df.to_csv(output_dir +'/test_df.csv',index = False)

# SMALL_SIZE = 7
# MEDIUM_SIZE = 7
# BIGGER_SIZE = 7

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('font', family='Nimbus Roman')

df_plot = train_df 
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10/3, 3))

# Iterate over label_code values
for label_code in range(4):
      dff = df_plot.query(f'label_code == {label_code}')
      out_array_train = []
      dd_train = np.array([out_array_train.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
      out_array_train = np.array(out_array_train)
      
      # Plot the KDE curve with a hue
      sns.kdeplot(out_array_train, label=f'{dic_labels_rev[label_code]}', ax=ax)

# Set the x-axis limits
ax.set_xlim(-1, 1)

# Customize the plot
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(False)
ax.legend()
plt.savefig(output_dir +'/Density_plot_train.png')
plt.savefig(output_dir +'/Density_plot_train.svg',format='svg', dpi=1200)
plt.close()


df_plot = test_df 
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10/3, 3))

# Iterate over label_code values
for label_code in range(4):
      dff = df_plot.query(f'label_code == {label_code}')
      out_array_train = []
      dd_train = np.array([out_array_train.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
      out_array_train = np.array(out_array_train)
      
      # Plot the KDE curve with a hue
      sns.kdeplot(out_array_train, label=f'{dic_labels_rev[label_code]}', ax=ax)

# Set the x-axis limits
ax.set_xlim(-1, 1)

# Customize the plot
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(False)
ax.legend()
plt.savefig(output_dir +'/Density_plot_test.png')
plt.savefig(output_dir +'/Density_plot_test.svg',format='svg', dpi=1200)
plt.close()


df_plot = valid_df 
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10/3, 3))

# Iterate over label_code values
for label_code in range(4):
      dff = df_plot.query(f'label_code == {label_code}')
      out_array_train = []
      dd_train = np.array([out_array_train.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
      out_array_train = np.array(out_array_train)
      
      # Plot the KDE curve with a hue
      sns.kdeplot(out_array_train, label=f'{dic_labels_rev[label_code]}', ax=ax)

# Set the x-axis limits
ax.set_xlim(-1, 1)

# Customize the plot
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(False)
ax.legend()
plt.savefig(output_dir +'/Density_plot_valid.png')
plt.savefig(output_dir +'/Density_plot_valid.svg',format='svg', dpi=1200)
plt.close()
#####################################################################
#####################################################################
#%%#
thresholds_abs_values = np.arange(-1, 1.2, 0.1)
mAP_results_valid = []
mAP_results_test = []
for _,thresh in enumerate(thresholds_abs_values):
      
      mAP_valid_thresh,_,_, _, _, _,_ = mAP_values(train_df,valid_df,thresh = thresh, percentile = False)
      mAP_test_thresh,_,_, _, _, _,_ = mAP_values(train_df, test_df,thresh = thresh, percentile = False)

      mAP_results_valid.append(mAP_valid_thresh)
      mAP_results_test.append(mAP_test_thresh)

      #topkmap, cum_prec, cum_recall, cum_map = MapWithPR_values(train_df,valid_df,thresh = thresh, percentile = False)
      

# Plotting
data_abs_values = {'mAP_valid': mAP_results_valid,
         'mAP_test': mAP_results_test,
         'threshold': thresholds_abs_values}

df_thresh_abs_values = pd.DataFrame(data_abs_values)
df_thresh_abs_values.to_csv(output_dir +'/results_data_abs_values.csv',index = False)

# Find the index of the maximum mAP value
max_map_index = df_thresh_abs_values['mAP_valid'].idxmax()

# Retrieve the threshold corresponding to the maximum mAP
threshold_max_map_abs_values = df_thresh_abs_values.loc[max_map_index, 'threshold']
#%%
mAP_valid_abs_values,mAP_std,mAP_values1, r_binary, train_label, q_binary, valid_label = mAP_values(train_df,valid_df,thresh = threshold_max_map_abs_values, percentile = False)
mAP_test_abs_values,mAP_std,mAP_values1, r_binary, train_label, q_binary, valid_label = mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False)


topkmap, cum_prec, cum_recall, cum_map = MapWithPR_values(train_df,valid_df,thresh = threshold_max_map_abs_values, percentile = False)


fig, ax = plt.subplots(figsize=(10/3, 3))

plt.plot(thresholds_abs_values, mAP_results_valid,color='#ff7f0eff')
plt.scatter(threshold_max_map_abs_values, mAP_valid_abs_values, color='#d62728ff', marker='o', s=25)
plt.xlabel('Threshold')
plt.ylabel('mAP')

#plt.rc('font', family='Nimbus Roman')

plt.savefig(output_dir + '/Maps_curves_abs_values.svg',format='svg', dpi=1200)
plt.show()

#%%
mAP_valid_abs_values,mAP_std,mAP_values1, r_binary, train_label, q_binary, valid_label = mAP_values(train_df,test_df,thresh = threshold_max_map_abs_values, percentile = False)
mAP, cum_prec, cum_recall,cum_map = MapWithPR_values(train_df,test_df,thresh = threshold_max_map_abs_values, percentile = False)

num_dataset=train_df.shape[0]
index_range = num_dataset // 100
index = [i * 100 - 1 for i in range(1, index_range + 1)]
max_index = max(index)
overflow = num_dataset - index_range * 100
index = index + [max_index + i for i in range(1, overflow + 1)]
#index = range(1, len(cum_prec))
c_prec = cum_prec[index]
c_recall = cum_recall[index]
#cum_map = cum_map[index]

pr_data = {
   "index": index,
   "P": c_prec.tolist(),
   "R": c_recall.tolist()
   #"M": cum_map.tolist()
}

with open(output_dir + '/cosfire_200.json', 'w') as f:
            
            f.write(json.dumps(pr_data))





# %%


pr_data = {
    "COSFIRE": output_dir + '/cosfire_200.json',
    "COSFIRE_500": output_dir + '/results.json'
   }
N = 100
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(15, 5))
plt.subplot(131)

for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    print(len(P))
    print(len(R))
    plt.plot(R, P, linestyle="-", label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.subplot(132)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, R, linestyle="-",  label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('recall')
plt.legend()

plt.subplot(133)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('precision')
plt.legend()
plt.savefig("pr.png")
plt.show()
# %%





#%%
SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
pr_data = {
	"COSFIRE": 'COSFIRE_1.json',
	"DenseNet": 'DenseNet.json'
}
N = 1200
# N = -1
for key in pr_data:
	path = pr_data[key]
	pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
	method2marker[method] = markers[i]
	i += 1

plt.figure(figsize=(15, 5))
plt.subplot(131)

for method in pr_data:
	P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
	print(len(P))
	print(len(R))
	plt.plot(R, P, linestyle="-",  label=method)
plt.grid(False)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.subplot(132)
for method in pr_data:
	P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
	plt.plot(draw_range, R, linestyle="-",  label=method)
plt.xlim(0, max(draw_range))
plt.grid(False)
plt.xlabel('The number of retrieved samples')
plt.ylabel('Recall')
plt.legend()

plt.subplot(133)
for method in pr_data:
	P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
	plt.plot(draw_range, P, linestyle="-",  label=method)
plt.xlim(0, max(draw_range))
plt.grid(False)
plt.xlabel('The number of retrieved samples')
plt.ylabel('Precision')
plt.legend()
plt.savefig('paper images/curves.svg',format='svg', dpi=1200)
plt.show()

# %%

SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

pr_data = {
	"COSFIRE": 'COSFIRE_1_raw.json',
	"DenseNet": 'DenseNet_raw.json'
}
N = 1200
# N = -1
for key in pr_data:
	path = pr_data[key]
	pr_data[key] = json.load(open(path))

# for method in pr_data:
# 	P, R,indices = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
     
mAP_cosfire = pr_data['COSFIRE']["mAP"]
mAP_densenet = pr_data['DenseNet']["mAP"]
R_cosfire = pr_data['COSFIRE']["R"]
R_densenet = pr_data["DenseNet"]["R"]
#R_indices = pr_data['COSFIRE']["index"]


P_cosfire = pr_data['COSFIRE']["P"]
P_densenet = pr_data["DenseNet"]["P"]
#P_indices = pr_data["DenseNet"]["index"]

######
#topkmap1234, cum_prec1234, cum_recall1234, cum_map1234
# Create the plot
plt.figure(figsize=(10, 6))

# Plot both curves
#plt.plot(R_cosfire,mAP_cosfire, label='COSFIRE', marker='o', markersize=3)
plt.plot(R_densenet,mAP_densenet, label='DenseNet', marker='s', markersize=3)
#plt.plot(R_densenet, P_densenet, label='DenseNet', marker='s', markersize=3)

# Customize the plot
plt.xlabel('Recall')
plt.ylabel('mAP')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()
#####
#%%

pr_data = {
	"COSFIRE": 'COSFIRE_1.json',
	"DenseNet": 'DenseNet.json'
}
N = 1200
# N = -1
for key in pr_data:
	path = pr_data[key]
	pr_data[key] = json.load(open(path))

for method in pr_data:
	P, R,indices = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
     
R_cosfire = pr_data['COSFIRE']["R"]
R_densenet = pr_data["DenseNet"]["R"]
R_indices = pr_data['COSFIRE']["index"]

P_cosfire = pr_data['COSFIRE']["P"]
P_densenet = pr_data["DenseNet"]["P"]
P_indices = pr_data["DenseNet"]["index"]

#%%
# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(R_cosfire,P_cosfire, label='COSFIRE', marker='o', markersize=3)
plt.plot(R_densenet, P_densenet, label='DenseNet', marker='s', markersize=3)

# Customize the plot
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig('paper images/precision_recall.svg',format='svg', dpi=1200)
plt.show()

#%%
# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, R_cosfire, label='COSFIRE', marker='o', markersize=3)
plt.plot(indices, R_densenet, label='DenseNet', marker='s', markersize=3)

# Customize the plot
plt.xlabel('The number of retrieved samples')
plt.ylabel('Recall')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig('paper images/recall_vs_images.svg',format='svg', dpi=1200)
plt.show()

#%%
# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, P_cosfire, label='Cosfire', marker='o', markersize=3)
plt.plot(indices, P_densenet, label='DenseNet', marker='s', markersize=3)

# Customize the plot
plt.xlabel('The number of retrieved samples')
plt.ylabel('Precision')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig('paper images/precision_vs_images.svg',format='svg', dpi=1200)
plt.show()
# %%
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# import matplotlib.font_manager
# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# sns.set_style("white")

# SMALL_SIZE = 7
# MEDIUM_SIZE = 7
# BIGGER_SIZE = 7

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# #plt.rc('figure', titlesize=BIGGER_SIZE)
# # Data
# # Create the plot
# fig, ax = plt.subplots(figsize=(10/3, 3))
# sns.set_style("whitegrid")
# plt.legend(loc = 'upper left')
# # Show the plot
# plt.rc('font', family='Nimbus Roman')

# df = pd.merge(result_merged_mwu_best, result_merged_mwu_std, on=['bit'])
# df = df.query('bit!=8')
# # Create line plot
# sns.lineplot(data=df, x='bit', y='test_score',label='Test',  color='blue')
# sns.lineplot(data=df, x='bit', y='valid_score',label='Valid',  color='red')
# # sns.lineplot(data=df, x='bit', y='valid_score_std',  color='blue')
# # sns.lineplot(data=df, x='bit', y='test_score_std',  color='red')
# # Fill standard deviation envelopes for 'test_score' and 'valid_score'
# plt.fill_between(df['bit'], df['test_score'] - df['test_score_std'], df['test_score'] + df['test_score_std'], color='#439CEF')
# plt.fill_between(df['bit'], df['valid_score'] - df['valid_score_std'], df['valid_score'] + df['valid_score_std'], color='#ffbaba')

# plt.scatter(72, df['test_score'].iloc[7], color='blue', marker='o', s=25)
# plt.scatter(72, df['valid_score'].iloc[7], color='red', marker='o', s=25)

# ax.set_ylabel('mAP')
# ax.set_xlabel('Bit size')
# ax = plt.gca()
# # Set y-axis to start from a specific value
# plt.ylim(85, 93)

# # Set x-axis labels and ticks
# x_labels = np.unique(np.array(df.bit))
# plt.xticks(ticks=x_labels, labels=x_labels)
# plt.grid(False)
# # Show the plot
# plt.tight_layout()
# plt.savefig('paper images/valid_test.svg',format='svg', dpi=1200)
# plt.show()



#%%
from sklearn.metrics.pairwise import cosine_similarity
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data_train = pd.read_csv("df_training_densenet.csv", sep = ',')
data_test = pd.read_csv("df_testing_densenet.csv", sep = ',' )

# train_df = pd.read_csv("FINAL_Best1_1/train_df.csv", sep = ',')
# test_df = pd.read_csv("FINAL_Best1_1/test_df.csv", sep = ',' )

def calculate_cosine_similarity(vector_a, vector_b):     
     vector_a = np.array(ast.literal_eval(vector_a))
     vector_b = np.array(ast.literal_eval(vector_b))
     # Reshape the vectors to 2D arrays because sklearn expects 2D arrays for pairwise metrics
     vector_a = vector_a.reshape(1, -1)
     vector_b = vector_b.reshape(1, -1)
     # Calculate the cosine similarity
     cosine_sim = cosine_similarity(vector_a, vector_b)
     return cosine_sim[0][0]

def calculate_cosine_similarity_v2(vector_a, vector_b):  
     # Reshape the vectors to 2D arrays because sklearn expects 2D arrays for pairwise metrics
     vector_a = vector_a.reshape(1, -1)
     vector_b = vector_b.reshape(1, -1)
     # Calculate the cosine similarity
     cosine_sim = cosine_similarity(vector_a, vector_b)
     return cosine_sim[0][0]

def calculate_norm_distance(vector_a, vector_b):
   vector_a = np.array(ast.literal_eval(vector_a))
   vector_b = np.array(ast.literal_eval(vector_b))
   dist = np.linalg.norm(vector_a - vector_b)
   return dist

def calculate_norm_distance_v2(vector_a, vector_b):
   dist = np.linalg.norm(vector_a - vector_b)
   return dist

def normalize_distance(arr):
   min_value = np.min(arr)
   max_value = np.max(arr)
   normalize_distance = (arr - min_value) / (max_value - min_value)
   return normalize_distance

# Convert numpy.float32 to float in the dictionary
def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
    


distances_densenet_dict = {}
distances_norm_densenet_dict = {}
distances_cosfire_dict = {}
distances_norm_cosfire_dict = {}

# Use tqdm to create a progress bar for the loop
for i in tqdm(range(data_test.shape[0]), desc="Processing Test Samples"):
    distances_densenet = []
    distances_norm_densenet = []
    distances_cosfire = []
    distances_norm_cosfire = []
    vector_a = data_test.predictions[i]
    vector_a_cosfire = test_df.predictions[i]
    for j in range(data_train.shape[0]):
        vector_b = data_train.predictions[j]
        distances_densenet.append(calculate_cosine_similarity(vector_a, vector_b))
        distances_norm_densenet.append(calculate_norm_distance(vector_a, vector_b))

        vector_b_cosfire = train_df.predictions[j]
        distances_cosfire.append(calculate_cosine_similarity_v2(vector_a_cosfire, vector_b_cosfire))
        distances_norm_cosfire.append(calculate_norm_distance_v2(vector_a_cosfire, vector_b_cosfire))
        

    # Convert lists to numpy arrays
    distances_densenet = np.array(distances_densenet)
    distances_norm_densenet = np.array(distances_norm_densenet)

    # Sort distances and store in dictionaries
    ind = np.argsort(distances_densenet)
    distances_densenet = distances_densenet[ind]
    distances_densenet_dict[i] = distances_densenet

    ind = np.argsort(distances_norm_densenet)
    distances_norm_densenet = distances_norm_densenet[ind]
    distances_norm_densenet_dict[i] = distances_norm_densenet

    # Convert lists to numpy arrays
    distances_cosfire = np.array(distances_cosfire)
    distances_norm_cosfire = np.array(distances_norm_cosfire)

    # Sort distances and store in dictionaries
    ind = np.argsort(distances_cosfire)
    distances_cosfire = distances_cosfire[ind]
    distances_cosfire_dict[i] = distances_cosfire

    ind = np.argsort(distances_norm_cosfire)
    distances_norm_cosfire = distances_norm_cosfire[ind]
    distances_norm_cosfire_dict[i] = distances_norm_cosfire
  

# distances_norm_densenet_dict_list={}
# distances_norm_cosfire_dict_list = {}
# for i in range(len(distances_norm_densenet_dict.keys())):
#     distances_norm_densenet_dict_list[i] = list(distances_norm_densenet_dict[i])  
#     distances_norm_cosfire_dict_list[i] = list(distances_norm_cosfire_dict[i])


# # Apply the conversion function to the dictionary
# distances_norm_cosfire_dict_list = convert_to_serializable(distances_norm_cosfire_dict_list)
# distances_norm_densenet_dict_list = convert_to_serializable(distances_norm_densenet_dict_list)
# with open(output_dir + '/distances_norm_densenet_dict_list.json', 'w') as json_file:
#     json.dump(distances_norm_densenet_dict_list, json_file)

# with open(output_dir + '/distances_norm_cosfire_dict_list.json', 'w') as json_file:
#     json.dump(distances_norm_cosfire_dict_list, json_file)


#json.load(open(output_dir + '/distances_norm_densenet_dict_list.json'))
#json.load(open(output_dir + '/distances_norm_cosfire_dict_list.json'))


#%%
# Data vectors

dat_norm_cosfire = distances_norm_cosfire_dict[0]
dat_norm_densenet = distances_norm_densenet_dict[0]
for i in range(1,len(distances_norm_cosfire_dict.keys())):
     dat_norm_cosfire =+ distances_norm_cosfire_dict[i]
     dat_norm_densenet =+ distances_norm_densenet_dict[i]
     
dat_cosfire_norm_average = dat_norm_cosfire/len(distances_norm_cosfire_dict.keys()) 
dat_densenet_norm_average = dat_norm_densenet/len(distances_norm_cosfire_dict.keys()) 


topk = 100
data1 = normalize_distance(dat_cosfire_norm_average[0:topk])

data2 = normalize_distance(dat_densenet_norm_average[0:topk])
# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(indices, data1, label='Cosfire norm', marker='o')
plt.plot(indices, data2, label='DenseNet norm', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images', fontsize=12)
plt.ylabel('Normalised distance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()


dat_cosfire = distances_cosfire_dict[0]
dat_densenet = distances_densenet_dict[0]
for i in range(1,len(distances_cosfire_dict.keys())):
     dat_cosfire =+ distances_cosfire_dict[i]
     dat_densenet =+ distances_densenet_dict[i]
     
dat_cosfire_average = dat_cosfire/len(distances_cosfire_dict.keys())
dat_densenet_average = dat_densenet/len(distances_densenet_dict.keys())

# Data vectors

data1 = normalize_distance(dat_cosfire_average[0:topk])
data2 = normalize_distance(dat_densenet_average[0:topk])

# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(indices, data1, label='Cosfire cosine', marker='o')
plt.plot(indices, data2, label='DenseNet cosine', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images', fontsize=12)
plt.ylabel('Normalised distance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig(output_dir + '/distances.png')
##############################################################
##############################################################
###########                                        ###########
##############################################################
##############################################################
# %%

distances_norm_densenet_dict_list = json.load(open('distances_norm_densenet_dict_list.json'))
distances_norm_cosfire_dict_list = json.load(open('distances_norm_cosfire_dict_list.json'))


dat_cosfire = np.array(distances_norm_cosfire_dict_list[str(0)])
dat_densenet = np.array(distances_norm_densenet_dict_list[str(0)])
for i in range(1,len(distances_norm_cosfire_dict_list.keys())):
     dat_cosfire =+ np.array(distances_norm_cosfire_dict_list[str(i)])
     dat_densenet =+ np.array(distances_norm_densenet_dict_list[str(i)])
     
dat_cosfire_average = dat_cosfire/len(distances_norm_cosfire_dict_list.keys())
dat_densenet_average = dat_densenet/len(distances_norm_densenet_dict_list.keys())

distances_cosine_densenet_dict_list = json.load(open('distances_cosine_densenet_dict.json'))
distances_cosine_cosfire_dict_list = json.load(open('distances_cosine_cosfire_dict_list.json'))

dat_cos_cosfire = np.array(distances_cosine_densenet_dict_list[str(0)])
dat_cos_densenet = np.array(distances_cosine_cosfire_dict_list[str(0)])
for i in range(1,len(distances_cosine_densenet_dict_list.keys())):
     dat_cos_cosfire =+ np.array(distances_cosine_densenet_dict_list[str(i)])
     dat_cos_densenet =+ np.array(distances_cosine_cosfire_dict_list[str(i)])
     
dat_cos_cosfire_average = dat_cos_cosfire/len(distances_cosine_densenet_dict_list.keys())
dat_cos_densenet_average = dat_cos_densenet/len(distances_cosine_cosfire_dict_list.keys())

#%%
topk =100
data1 = normalize_distance(dat_cosfire_norm_average[0:topk])

data2 = normalize_distance(dat_densenet_norm_average[0:topk])
# Create indices
indices = np.arange(len(data1))

SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, data1, label='Cosfire norm', marker='o')
plt.plot(indices, data2, label='DenseNet norm', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images')
plt.ylabel('Normalised distance')
plt.legend()
plt.grid(False, linestyle='--')

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()


# Data vectors

data1 = normalize_distance(dat_cos_cosfire_average[0:topk])
data2 = normalize_distance(dat_cos_densenet_average[0:topk])

# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, data1, label='Cosfire cosine', marker='o')
plt.plot(indices, data2, label='DenseNet cosine', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images')
plt.ylabel('Normalised distance')
plt.legend()
plt.grid(False, linestyle='--')

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()

# %%

##############################################################
##############################################################
###########                                        ###########
##############################################################
##############################################################
# %%

# test_df.label_name.value_counts()

start = 0
topk = 103

distances_norm_densenet_dict_list = json.load(open('distances_norm_densenet_dict_list.json'))
distances_norm_cosfire_dict_list = json.load(open('distances_norm_cosfire_dict_list.json'))


dat_cosfire = np.array(distances_norm_cosfire_dict_list[str(start)])
dat_densenet = np.array(distances_norm_densenet_dict_list[str(start)])
for i in range(start,topk):#len(distances_norm_cosfire_dict_list.keys())):
     dat_cosfire =+ np.array(distances_norm_cosfire_dict_list[str(i)])
     dat_densenet =+ np.array(distances_norm_densenet_dict_list[str(i)])
     
dat_cosfire_average = dat_cosfire/(topk-start)#len(distances_norm_cosfire_dict_list.keys())
dat_densenet_average = dat_densenet/(topk-start)#len(distances_norm_densenet_dict_list.keys())

distances_cosine_densenet_dict_list = json.load(open('distances_cosine_densenet_dict.json'))
distances_cosine_cosfire_dict_list = json.load(open('distances_cosine_cosfire_dict_list.json'))

dat_cos_cosfire = np.array(distances_cosine_densenet_dict_list[str(start)])
dat_cos_densenet = np.array(distances_cosine_cosfire_dict_list[str(start)])
for i in range(start,topk):#len(distances_cosine_densenet_dict_list.keys())):
     dat_cos_cosfire =+ np.array(distances_cosine_densenet_dict_list[str(i)])
     dat_cos_densenet =+ np.array(distances_cosine_cosfire_dict_list[str(i)])
     
dat_cos_cosfire_average = dat_cos_cosfire/(topk-start)#len(distances_cosine_densenet_dict_list.keys())
dat_cos_densenet_average = dat_cos_densenet/(topk-start)#len(distances_cosine_cosfire_dict_list.keys())


topk=(topk-start)
data1 = normalize_distance(dat_cosfire_norm_average[0:topk])

data2 = normalize_distance(dat_densenet_norm_average[0:topk])
# Create indices
indices = np.arange(len(data1))

SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, data1, label='Cosfire norm', marker='o', markersize=3)
plt.plot(indices, data2, label='DenseNet norm', marker='s', markersize=3)

# Customize the plot
plt.xlabel(f'Top {topk} images')
plt.ylabel('Normalised distance')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()


# Data vectors

data1 = normalize_distance(dat_cos_cosfire_average[0:topk])
data2 = normalize_distance(dat_cos_densenet_average[0:topk])

# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, data1, label='Cosfire', marker='o', markersize=3)
plt.plot(indices, data2, label='DenseNet', marker='s', markersize=3)

# Customize the plot
plt.xlabel(f'Top {topk} images')
plt.ylabel('Hamming distance')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig('paper images/distance_Bent.svg',format='svg', dpi=1200)
plt.show()
# %%


#%%



# %%
# topk_number_images = list(range(90,1180,10)) + [1180]
# mAP_tok = []
# for _, topk in enumerate(topk_number_images):
#    maP,r_binary, _, _, _ = mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False)
#    mAP_tok.append(maP)


# data = {'topk_number_images': topk_number_images,
#         'mAP': mAP_tok}
# df = pd.DataFrame(data)
# df.to_csv(f'mAP_vs_topk_images_72bit_cosfire.csv', index = False)

# # Plot the line curve
# sns.lineplot(x='topk_number_images', y='mAP', data=df, color = 'r')
# plt.xlabel('Top k number of images')
# plt.ylabel('mAP')
# plt.show()
# %%

dt_cosfire = pd.read_csv('mAP_vs_topk_images_72bit_cosfire.csv')
dt_densenet = pd.read_csv('mAP_vs_topk_images_72bit_densenet.csv')

num_dataset=dt_cosfire.shape[0]
index_range = num_dataset // 10
index = [i * 10 - 1 for i in range(1, index_range + 1)]
max_index = max(index)
overflow = num_dataset - index_range * 10
index = index + [max_index + i for i in range(1, overflow + 1)]
index = [0]+index
map_cosfire = dt_cosfire.mAP
map_densenet = dt_densenet.mAP
indices = dt_densenet.topk_number_images

map_cosfire = map_cosfire[index]
map_densenet = map_densenet[index]
indices = indices[index]

# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, map_cosfire, label='COSFIRE', marker='o', markersize=3)
plt.plot(indices, map_densenet, label='DenseNet', marker='s', markersize=3)

# Customize the plot
plt.xlabel(f'The number of retrieved samples')
plt.ylabel('mAP')
plt.legend()
plt.grid(False)
plt.ylim(80, 93)

# Show the plot
plt.tight_layout()
plt.savefig('paper images/mAP_vs_top_images.svg',format='svg', dpi=1200)
plt.show()
# %%



#%%

import numpy as np
vector_a = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1,
1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])
vector_b = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1,
1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])
def similarity_distance_count(vector_a, vector_b):
   similarity_distance = np.count_nonzero(vector_a!=vector_b)
   return similarity_distance

similarity_distance_count(vector_a, vector_b)



#distances_hamm_densenet_dict = {}
distances_hamm_cosfire_dict = {}

# Use tqdm to create a progress bar for the loop
for i in tqdm(range(test_binary.shape[0]), desc="Processing Test Samples"):
    distances_hamm_cosfire = []
    vector_a = test_binary[i]
    for j in range(train_binary.shape[0]):
        vector_b = train_binary[j]
        distances_hamm_cosfire.append(similarity_distance_count(vector_a, vector_b))

    distances_hamm_cosfire = np.array(distances_hamm_cosfire)
    
    ind = np.argsort(distances_hamm_cosfire)
    distances_hamm_cosfire = distances_hamm_cosfire[ind]
    distances_hamm_cosfire_dict[i] = distances_hamm_cosfire


distances_hamm_cosfire_dict_list = {}
for i in range(len(distances_hamm_cosfire_dict.keys())):

    distances_hamm_cosfire_dict_list[i] = list(distances_hamm_cosfire_dict[i])


# Apply the conversion function to the entire dictionary
converted_dict = convert_numpy(distances_hamm_cosfire_dict_list)

# Now save the converted dictionary as JSON
with open(output_dir + '/distances_hamm_cosfire_dict_list.json', 'w') as json_file:
    json.dump(distances_hamm_cosfire_dict_list, json_file)



dat_hamm_cosfire = np.array(distances_hamm_cosfire_dict[0])

for i in range(1,len(distances_hamm_cosfire_dict.keys())):
    dat_hamm_cosfire = dat_hamm_cosfire + np.array(distances_hamm_cosfire_dict[i])
    
    
dat_cosfire_hamm_average = dat_hamm_cosfire/len(distances_hamm_cosfire_dict.keys()) 


  

topk = 100
data1 = dat_cosfire_hamm_average[0:topk]

#data2 = dat_densenet_hamm_average[0:topk]
# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(indices, data1, label='Cosfire hamm', marker='o')
#plt.plot(indices, data2, label='DenseNet hamm', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images', fontsize=12)
plt.ylabel('hammalised distance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig(output_dir + "/distance_measure_hamm_comparison.png")
plt.show()
plt.close()


dat_cosfire = distances_hamm_cosfire_dict[0]
#dat_densenet = distances_densenet_dict[0]
for i in range(1,len(distances_hamm_cosfire_dict.keys())):
    dat_cosfire =+ distances_hamm_cosfire_dict[i]
    #dat_densenet =+ distances_densenet_dict[i]
    
dat_cosfire_average = dat_cosfire/len(distances_hamm_cosfire_dict.keys())
#dat_densenet_average = dat_densenet/len(distances_densenet_dict.keys())

# Data vectors

data1 = dat_cosfire_average[0:topk]
#data2 = dat_densenet_average[0:topk]

# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(indices, data1, label='Cosfire cosine', marker='o')
#plt.plot(indices, data2, label='DenseNet cosine', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images', fontsize=12)
plt.ylabel('Hamming distance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig(output_dir + "/distance_measure_hamm_comparison.png")
plt.show()
plt.close()


#%%

train_df_dn = pd.read_csv('df_training_tl.csv')
test_df_dn = pd.read_csv('df_testing_tl.csv')

predictions_test = []
predictions_train = []

for i in range(train_df_dn.shape[0]):
    predictions_train.append(np.array(ast.literal_eval(train_df_dn.predictions[i])))

for i in range(test_df_dn.shape[0]):
    predictions_test.append(np.array(ast.literal_eval(test_df_dn.predictions[i])))

train_df_dn['predictions'] = predictions_train
test_df_dn['predictions'] = predictions_test


# %%
threshold_max_map_abs_values = -0.9
mAP,mAP_std, r_binary, train_label, q_binary, valid_label= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
rB = r_binary
qB = q_binary
retrievalL = train_label
queryL = valid_label

topk=1180
mAPs = np.zeros(404)
mAPs_stds = np.zeros(404)
for i, query in enumerate(queryL):
    rel = (np.dot(query, retrievalL.transpose()) > 0)*1 # relevant train label refs.
    hamm = np.count_nonzero(qB[i] != rB, axis=1) #hamming distance
    ind = np.argsort(hamm) #  Hamming distances in ascending order.
    rel = rel[ind] #rel is reordered based on the sorted indices ind, so that it corresponds to the sorted Hamming distances.

    top_rel = rel[:topk] #contains the relevance values for the top-k retrieved results
    tot_relevant_sum = np.sum(top_rel) #total number of relevant images in the top-k retrieved results

    #skips the iteration if there are no relevant results.
    if tot_relevant_sum == 0:
        continue

    pr_num = np.linspace(1, tot_relevant_sum, tot_relevant_sum) 
    pr_denom = np.asarray(np.where(top_rel == 1)) + 1.0 #is the indices where top_rel is equal to 1 (i.e., the positions of relevant images)
    pr = pr_num / pr_denom[0] # precision
    mAP_sub = np.mean(pr) # AP - The average is based on GTP since we are taking tsum    
    mAP_sub_std = np.std(pr)
    mAPs[i] = mAP_sub 
    mAPs_stds[i] = mAP_sub_std
    break

    
    

#%%
data1 = pr_num
data2 = pr_denom[0]

#data2 = dat_densenet_hamm_average[0:topk]
# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(data1, data2, label='pr_num')
#plt.plot(indices, data2, label='pr_denom')
#plt.plot(indices, pr[0], label='precision')

# Customize the plot
plt.xlabel('Numerator', fontsize=12)
plt.ylabel('Denominator', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis

# Show the plot
plt.tight_layout()
plt.show()

# %%


#data2 = dat_densenet_hamm_average[0:topk]
# Create indices
indices = np.arange(len(pr_num / pr_denom[0]))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(indices, pr_num / pr_denom[0], label='precision')

# Customize the plot
plt.xlabel('index', fontsize=12)
plt.ylabel('prec values', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis

# Show the plot
plt.tight_layout()
plt.show()
# %%





#%%
label = 'mean_std_George'
train_df_dn = pd.read_csv('df_training_tl.csv')
test_df_dn = pd.read_csv('df_testing_tl.csv')

predictions_test = []
predictions_train = []

for i in range(train_df_dn.shape[0]):
    predictions_train.append(np.array(ast.literal_eval(train_df_dn.predictions[i])))

for i in range(test_df_dn.shape[0]):
    predictions_test.append(np.array(ast.literal_eval(test_df_dn.predictions[i])))

train_df_dn['predictions'] = predictions_train
test_df_dn['predictions'] = predictions_test

threshold_max_map_abs_values = -0.9
topk=100
mAP_dn,mAP_std_dn,mAP_values_dn, r_binary_dn, train_label_dn, q_binary_dn, valid_label_dn = mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
print('mAP top 100  for DenseNet is: ',np.round(mAP_dn,2))

topk_number_images_dn = list(range(10,1180,50)) + [1180]
mAP_topk_dn = []
mAP_topk_std_dn = []
map_values_list = []
for _, topk in enumerate(topk_number_images_dn):
    maP_dn,mAP_std_dn,map_values, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    mAP_topk_dn.append(maP_dn)
    mAP_topk_std_dn.append(mAP_std_dn)
    map_values_list.append(map_values)
    

data_densenet = {'topk_number_images': topk_number_images_dn,
        'mAP': mAP_topk_dn,
        'mAP_std': mAP_topk_std_dn}
df_densenet = pd.DataFrame(data_densenet)
df_densenet.to_csv(f'mAP_vs_topk_images_72bit_densenet_{label}.csv', index = False)

# Plot the line curve
sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet',  color='blue')
plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#439CEF')
plt.xlabel('The number of retrieved samples')
plt.ylabel('mAP')
plt.ylim(0, 103)
plt.savefig( f'map_vs_topk_number_images_densenet_{label}.png')
plt.savefig( f'map_vs_topk_number_images_densenet_{label}.svg',format='svg', dpi=1200)
plt.show()



threshold_max_map_abs_values = -0.9
topk=100
mAP,mAP_std,mAP_values1, r_binary, train_label, q_binary, valid_label= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
print('mAP top 100  for COSFIRE is: ',np.round(mAP,2))

topk_number_images = list(range(10,1180,50)) + [1180]
mAP_topk = []
mAP_topk_std = []
map_values_list = []
for _, topk in enumerate(topk_number_images):
    maP,mAP_std,map_values, _, _, _,_= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    mAP_topk.append(maP)
    mAP_topk_std.append(mAP_std)
    map_values_list.append(map_values)

data = {'topk_number_images': topk_number_images,
        'mAP': mAP_topk,
        'mAP_std': mAP_topk_std}
df_cosfire = pd.DataFrame(data)
df_cosfire.to_csv('mAP_vs_topk_images_72bit_cosfire_{label}.csv', index = False)

sns.lineplot(data=df_cosfire, x='topk_number_images', y='mAP',label='Cosfire',  color='blue')
plt.fill_between(df_cosfire['topk_number_images'], df_cosfire['mAP'] - df_cosfire['mAP_std'], df_cosfire['mAP'] + df_cosfire['mAP_std'], color='#439CEF')
plt.xlabel('The number of retrieved samples')
plt.ylabel('mAP')
plt.ylim(0, 110)
plt.savefig( f'map_vs_topk_number_images_cosfire_{label}.png')
plt.savefig( f'map_vs_topk_number_images_cosfire_{label}.svg',format='svg', dpi=1200)
plt.show()


# %%

#fig, ax = plt.subplots(figsize=(10/3, 3))
sns.lineplot(data=df_cosfire, x='topk_number_images', y='mAP',label='Cosfire',  color='red')
# plt.fill_between(df_cosfire['topk_number_images'], df_cosfire['mAP'] - df_cosfire['mAP_std'], df_cosfire['mAP'] + df_cosfire['mAP_std'], color='#439CEF')
sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet',  color='blue')
# plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#ffbaba')

plt.xlabel('The number of retrieved samples')
plt.ylabel('mAP')
plt.ylim(80, 95)
plt.savefig( f'map_vs_topk_number_images_densenet_{label}_cbd1.png')
plt.savefig( f'map_vs_topk_number_images_densenet_{label}_cbd1.svg',format='svg', dpi=1200)
plt.show()
# %%


#%%
# # When R=k Special case:
# if (query == np.array([0, 0, 1, 0])).sum()==4:#Bent
#         mAP_sub = np.sum(pr) /np.min(np.array([305,topk]))
# elif (query == np.array([0, 1, 0, 0])).sum()==4:#FRII
#         mAP_sub = np.sum(pr) / np.min(np.array([434,topk]))
# elif (query == np.array([1, 0, 0, 0])).sum()==4:#FRI
#         mAP_sub = np.sum(pr) /  np.min(np.array([215,topk]))
# else:# (query == np.array([0, 0, 0, 1])).sum()==4:#Compact
#         mAP_sub = np.sum(pr) / np.min(np.array([226,topk]))
threshold_max_map_abs_values = -0.9
#Bent
topk=305
mAP_bent,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#FRII
topk=434
mAP_frii,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#FRI
topk=215
mAP_fri,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#Compact
topk=226
mAP_comp,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)

print('====== COSFIRE ======')
print('mAP (305) Bent :',round(mAP_bent,2))
print('mAP (434) FRII :',round(mAP_frii,2))
print('mAP (215) FRI :',round(mAP_fri,2))
print('mAP (226) Compact :',round(mAP_comp,2))
print('Average mAP when R=K: ',round(np.mean(np.array([mAP_bent,mAP_frii,mAP_fri,mAP_comp])),2))
print('====== COSFIRE ======\n')

def mAP_at_k_equals_R(train_df, test_df,threshold_max_map_abs_values,topk):        
    #Bent
    topk=305
    mAP_bent,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    #FRII
    topk=434
    mAP_frii,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    #FRI
    topk=215
    mAP_fri,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    #Compact
    topk=226
    mAP_comp,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    mean_average = round(np.mean(np.array([mAP_bent,mAP_frii,mAP_fri,mAP_comp])),2)
    
    return mean_average


threshold_max_map_abs_values = -0.9
#Bent
topk=305
mAP_bent_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#FRII
topk=434
mAP_frii_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#FRI
topk=215
mAP_fri_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#Compact
topk=226
mAP_comp_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)

print('====== DenseNet ======')
print('mAP (305) Bent :',round(mAP_bent_dn,2))
print('mAP (434) FRII :',round(mAP_frii_dn,2))
print('mAP (215) FRI :',round(mAP_fri_dn,2))
print('mAP (226) Compact :',round(mAP_comp_dn,2))
print('Average mAP when R=K: ',round(np.mean(np.array([mAP_bent_dn,mAP_frii_dn,mAP_fri_dn,mAP_comp_dn])),2))
print('====== DenseNet ======\n')
# %%
