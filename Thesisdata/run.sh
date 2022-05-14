device=0
data=/content/drive/MyDrive/Thesisdata/mydata/
batch=1
n_head=6
n_layers=4
d_model=512
d_inner=1024
d_k=512
d_v=512
dropout=0.1
lr=1e-4
smooth=0.1
epoch=100
log=/content/drive/MyDrive/Thesisdata/mydata/log.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python /content/drive/MyDrive/Thesisdata/Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log
