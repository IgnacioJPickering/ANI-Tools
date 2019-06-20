import anialservertools as ast
import anialtools as alt
import anitraintools as att
import os


# Where the calculations are run in the external server
hostname = "moria.chem.ufl.edu"
username = "ipickering"
swork_dir = '/home/ipickering/scratch/auto_al_cycles/'
jtime = "0-6:00"
mae = ''
#potentially a bunch of work_dir inside model_dir, each of them 
# corresponds to a "set" of cycles
# for each set of cycles inputtrain, sae_linfit and rHCNO are all THE SAME
# so the are all inside model_dir directly
# people manually change work_dir each time
root_dir = '/home/ipickering/scratch/auto_dhl_al/'
# Storage location for all the optimized structures
optstruct_file = root_dir + 'optimized_input_files.dat'
# Storage location for h5files
h5_dir = root_dir + 'h5files/'
datdir = 'ANI-1x-DHL-0000.00'
model_dir =  root_dir + 'modeldhl/'
work_dir = model_dir + 'ANI-1x-DHL-0000/'
ipt_file = model_dir + 'inputtrain.ipt'
sae_file = model_dir + 'sae_linfit.dat'
cst_file = model_dir + 'rHCNO-5.2R_16-3.5A_a4-8.params'
fpatoms = ['C', 'N', 'O']
# Training Parameters 
#gpu_idlist is a list that holds all the GPU id's
gpu_idlist = [0, 1]
M   = 0.35 # Max error per atom in kcal/mol
# The dataset is butchered into Nblock blocks, where Nblocks is a multiple of 
# Nnets. 
# From these blocks Nstrides-1 "nodes" are setup, spaced with spacing Nstrides
# The test and validation sets for each of the networks in the ensemble
# are set up so that they overlap "as little as possible", each T+V set
# starting from one of the Nstrides-1 nodes.
# Nblock, Nbvald, Nbtest are the number of blocks in total, in the validation
# set and in the test set respectively. Nbtrain is calculated to be 
# Nbtrain  = Nblock - (Nbvald + Nbtest), the blocks that are left over
Nnets = 8 
Nblock = 16 
Nbvald = 3 
Nbtest = 1 
#aevsize is the size of the input to the ANIModule.
aev_size = 384
# Sampling parameters 
dhparams = {'Nmol': 100, 'Nsamp': 4, 'sig' : 0.08, 'rng' : 0.2, 'MaxNa' : 40,
             'smilefile': root_dir + 'dhl_files/dhl_genentech.smi'}
             # 'smilefile': 
             # '/home/jsmith48/scratch/auto_dhl_al/dhl_files/dhl_genentech.smi'}
# Active learning cycles
cycles = range(20)
for cycle in cycles:
    # network_dir is the directory where each of the networks generated in each 
    # cycle is stored
    network_dir = work_dir + datdir + str(cycle).zfill(2) + '/'
    cache_dir = network_dir + "cache-data-"
    # directory for the active learning daa
    # this used to be cycle+1 for some reason, I just made it cycle
    aldata_dir = root_dir + datdir + str(cycle).zfill(2)
    nnfprefix   = network_dir + 'train'
    netdict = {
            'iptfile' : ipt_file, 'cnstfile' : cst_file, 'saefile': sae_file,
            'nnfprefix': network_dir+'train', 'aevsize': aev_size, 'num_nets': Nnets,
            'atomtyp' : ['H','C','N','O']}
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print('Had to create cache_dir directory')
    if not os.path.exists(network_dir):
        os.makedirs(network_dir)
        print('Had to create network_dir directory')
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)
        print('Had to create h5_dir directory')
    if not os.path.exists(aldata_dir):
        os.makedirs(aldata_dir)
        print('Had to create this strange long directory')
    try:
        assert os.listdir(h5_dir)
    except AssertionError:
        raise AssertionError('Your h5_dir directory is empty, add some files!')

    ########## Training the ensemble #############
    #This gets network_dir = work_dir + datdir(withstuff)
    inputbuilder = att.anitrainerinputdesigner()
    aet = att.alaniensembletrainer(network_dir, netdict, inputbuilder, h5_dir, Nnets)
    aet.build_strided_training_cache(Nblock,Nbvald,Nbtest,False)
    aet.train_ensemble(gpu_idlist)
    ##############################################

    ############### Run active learning ################
    # This gets root_dir + datdir(withstuff)
    # alconformationalsampler combines root_dir and datdir(withstuff) in 
    # all calls
    acs = alt.alconformationalsampler(
            root_dir, datdir + str(cycle+1).zfill(2),
            optstruct_file, fpatoms, netdict)
    acs.run_sampling_dhl(dhparams, gpus=gpu_idlist+gpu_idlist)
    ####################################################

    ######### Submit jobs, return and pack data ##########
    # This gets root_dir + datdir(withstuff)
    ast.generateQMdata(
            hostname, username, swork_dir,
            root_dir, datdir + str(cycle+1).zfill(2), h5_dir, mae, jtime)
    ###################################################
