import anialservertools as aniserver
import anialtools as alt
import os


root_dir = '/home/jsmith48/scratch/auto_dhl_al/'
h5dataset_path = os.path.join(root_dir,'h5files')
model_path = os.path.join(root_dir,'modeldhl')
wkdir_path = os.path.join(model_path,'ANI-1x-DHL-0000')
iptfile_path = os.path.join(model_path,'inputtrain.ipt')
saefile_path = os.path.join(model_path,'sae_linfit.dat')
cstfile_path = os.path.join(model_path,'rHCNO-5.2R_16-3.5A_a4-8.params')
optlfile_path = root_dir + 'optimized_input_files.dat'

#Server parameters
swkdir = '/home/smith48/scratch/auto_al_cycles/'# server working directory
mae = 'module load gnu/4.9.2\n' +\
      'module load gaussian\n' +\
      'export PATH="/home/$USER/Gits/RCDBuilder/build/bin:$PATH"\n' +\
      'export LD_LIBRARY_PATH="/home/$USER/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n'
jtime = "0-6:00"
hostname = "comet.sdsc.xsede.org"
username = "jsmith48"

fpatoms = ['C', 'N', 'O']
#---- Training Parameters ----
GPU = [0, 7] # GPU IDs
Nnets = 8 # networks in ensemble
Nblock = 16 # Number of blocks in split
Nbvald = 3 # number of valid blocks
Nbtest = 1 # number of test blocks
aevsize = 384

# Training varibles
dhparams = { 'Nmol': 100, 'Nsamp': 4, 'sig' : 0.08, 'rng' : 0.2, 'MaxNa' : 40,
             'smilefile': '/home/jsmith48/scratch/auto_dhl_al/dhl_files/dhl_genentech.smi',
             }
# Begin active learning cycles
active_learning_cycles = 10
for i in range(active_learning_cycles):
    dat = 'ANI-1x-DHL-0000.00' + str(i).zfill(2)
    netdir = os.path.join(wkdir_path,dat)
    new_datdir = os.path.join(root_dir,dat)
    nnfprefix = netdir + 'train'
    try:
        os.mkdir(netdir)
    except FileExistsError:
        pass
    try:
        os.mkdir(new_datdir)
    except FileExistsError:
        pass

    netdict = {'iptfile' : iptfile_path,
               'cnstfile' : cstfile_path,
               'saefile' : saefile_path,
               'nnfprefix': nnfprefix,
               'aevsize': aevsize,
               'num_nets': Nnets,
               'atomtyp' : ['H','C','N','O']}
    # Train the ensemble 
    ani_ensemble_trainer = alt.alaniensembletrainer(
            netdir, netdict, h5dataset_path, Nnets)
    ani_ensemble_trainer.build_strided_training_cache(
            Nblock, Nbvald, Nbtest, False)
    ani_ensemble_trainer.train_ensemble(GPU)
    local_data_dir = root_dir
    # Run active learning sampling 
    ani_conformational_sampler = alt.alconformationalsampler(local_data_dir, 
            dat, optlfile_path, fpatoms, netdict)
    ani_conformational_sampler.run_sampling_dhl(dhparams, gpus=GPU+GPU)
    # Submit jobs, return and pack data
    aniserver.generateQMdata(hostname, username, swkdir, local_data_dir, dat, 
            h5dataset_path, mae, jtime)
