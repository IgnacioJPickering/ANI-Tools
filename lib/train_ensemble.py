class alaniensembletrainer():
    def __init__(self, train_root, netdict, input_builder, h5dir, Nn, random_seed=-1):
        if random_seed != -1:
            np.random.seed(random_seed)
        self.train_root = train_root
        self.h5dir = h5dir
        self.Nn = Nn
        self.netdict = netdict
        self.iptbuilder = input_builder
        if h5dir is not None:
            self.h5file = [f for f in os.listdir(self.h5dir) if f.rsplit('.', 1)[1] == 'h5']
    def build_training_cache(self, forces=True):
        store_dir = self.train_root + "cache-data-"
        N = self.Nn
        for i in range(N):
            if not os.path.exists(store_dir + str(i)):
                os.mkdir(store_dir + str(i))
            if os.path.exists(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5'):
                os.remove(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5')
            if not os.path.exists(store_dir + str(i) + '/../testset'):
                os.mkdir(store_dir + str(i) + '/../testset')
        cachet = [cg('_train', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        cachev = [cg('_valid', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        testh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/testset' + str(r) + '.h5') for r in range(N)]
        Nd = np.zeros(N, dtype=np.int32)
        Nbf = 0
        for f, fn in enumerate(self.h5file):
            print('Processing file(' + str(f + 1) + ' of ' + str(len(self.h5file)) + '):', fn)
            adl = pyt.anidataloader(self.h5dir + fn)
            To = adl.size()
            Ndc = 0
            Fmt = []
            Emt = []
            for c, data in enumerate(adl):
                Pn = data['path'] + '_' + str(f).zfill(6) + '_' + str(c).zfill(6)
                # Extract the data
                X = data['coordinates']
                E = data['energies']
                S = data['species']
                # 0.0 forces if key doesnt exist
                if forces:
                    F = data['forces']
                else:
                    F = 0.0*X
                Fmt.append(np.max(np.linalg.norm(F, axis=2), axis=1))
                Emt.append(E)
                Mv = np.max(np.linalg.norm(F, axis=2), axis=1)
                index = np.where(Mv > 10.5)[0]
                indexk = np.where(Mv <= 10.5)[0]
                Nbf += index.size
                # CLear forces
                X = X[indexk]
                F = F[indexk]
                E = E[indexk]
                Esae = hdt.compute_sae(self.netdict['saefile'], S)
                hidx = np.where(np.abs(E - Esae) > 10.0)
                lidx = np.where(np.abs(E - Esae) <= 10.0)
                if hidx[0].size > 0:
                    print('  -(' + str(c).zfill(3) + ')High energies detected:\n    ', E[hidx])
                X = X[lidx]
                E = E[lidx]
                F = F[lidx]
                Ndc += E.size
                if (set(S).issubset(self.netdict['atomtyp'])):
                    # Random mask
                    R = np.random.uniform(0.0, 1.0, E.shape[0])
                    idx = np.array([interval(r, N) for r in R])
                    # Build random split lists
                    split = []
                    for j in range(N):
                        split.append([i for i, s in enumerate(idx) if s == j])
                        nd = len([i for i, s in enumerate(idx) if s == j])
                        Nd[j] = Nd[j] + nd
                    # Store data
                    for i, t, v, te in zip(range(N), cachet, cachev, testh5):
                        ## Store training data
                        X_t = np.array(np.concatenate([X[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float32)
                        F_t = np.array(np.concatenate([F[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float32)
                        E_t = np.array(np.concatenate([E[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float64)
                        if E_t.shape[0] != 0:
                            t.insertdata(X_t, F_t, E_t, list(S))
                        ## Store Validation
                        if np.array(split[i]).size > 0:
                            X_v = np.array(X[split[i]], order='C', dtype=np.float32)
                            F_v = np.array(F[split[i]], order='C', dtype=np.float32)
                            E_v = np.array(E[split[i]], order='C', dtype=np.float64)
                            if E_v.shape[0] != 0:
                                v.insertdata(X_v, F_v, E_v, list(S))
        # Print some stats
        print('Data count:', Nd)
        print('Data split:', 100.0 * Nd / np.sum(Nd), '%')
        # Save train and valid meta file and cleanup testh5
        for t, v, th in zip(cachet, cachev, testh5):
            t.makemetadata()
            v.makemetadata()
            th.cleanup()

    def sae_linear_fitting(self, Ekey='energies', energy_unit=1.0, Eax0sum=False):
        from sklearn import linear_model
        print('Performing linear fitting...')
        datadir = self.h5dir
        sae_out = self.netdict['saefile']
        smap = dict()
        for i, Z in enumerate(self.netdict['atomtyp']):
            smap.update({Z: i})
        Na = len(smap)
        files = os.listdir(datadir)
        X = []
        y = []
        for f in files[0:20]:
            print(f)
            adl = pyt.anidataloader(datadir + f)
            for data in adl:
                S = data['species']
                if data[Ekey].size > 0:
                    if Eax0sum:
                        E = energy_unit*np.sum(np.array(data[Ekey], order='C', dtype=np.float64), axis=1)
                    else:
                        E = energy_unit*np.array(data[Ekey], order='C', dtype=np.float64)
                    S = S[0:data['coordinates'].shape[1]]
                    unique, counts = np.unique(S, return_counts=True)
                    x = np.zeros(Na, dtype=np.float64)
                    for u, c in zip(unique, counts):
                        x[smap[u]] = c
                    for e in E:
                        X.append(np.array(x))
                        y.append(np.array(e))
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        lin = linear_model.LinearRegression(fit_intercept=False)
        lin.fit(X, y)
        coef = lin.coef_
        print(coef)
        sae = open(sae_out, 'w')
        for i, c in enumerate(coef[0]):
            sae.write(next(key for key, value in smap.items() if value == i) + ',' + str(i) + '=' + str(c) + '\n')
        sae.close()
        print('Linear fitting complete.')

    def build_strided_training_cache(self, Nblocks, Nvalid, Ntest, build_test=True,
                                     Ekey='energies', energy_unit=1.0,
                                     forces=True, grad=False, Fkey='forces', forces_unit=1.0,
                                     dipole=False, dipole_unit=1.0, Dkey='dipoles',
                                     charge=False, charge_unit=1.0, Ckey='charges',
                                     pbc=False,
                                     Eax0sum=False, rmhighe=True,rmhighf=False,force_exact_split=False):
        if not os.path.isfile(self.netdict['saefile']):
            self.sae_linear_fitting(Ekey=Ekey, energy_unit=energy_unit, Eax0sum=Eax0sum)
        h5d = self.h5dir
        store_dir = self.train_root + "cache-data-"
        N = self.Nn
        Ntrain = Nblocks - Nvalid - Ntest
        if Nblocks % N != 0:
            raise ValueError('Error: number of networks must evenly divide number of blocks.')
        Nstride = Nblocks/N
        for i in range(N):
            if not os.path.exists(store_dir + str(i)):
                os.mkdir(store_dir + str(i))
            if build_test:
                if os.path.exists(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5'):
                    os.remove(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5')
                if not os.path.exists(store_dir + str(i) + '/../testset'):
                    os.mkdir(store_dir + str(i) + '/../testset')
        cachet = [cg('_train', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        cachev = [cg('_valid', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        if build_test:
            testh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/testset' + str(r) + '.h5') for r in range(N)]


        if rmhighe:
            dE = []
            for f in self.h5file:
                adl = pyt.anidataloader(h5d + f)
                for data in adl:
                    S = data['species']
                    E = data[Ekey]
                    X = data['coordinates']
                    Esae = hdt.compute_sae(self.netdict['saefile'], S)
                    dE.append((E - Esae)/np.sqrt(len(S)))
            dE = np.concatenate(dE)
            cidx = np.where(np.abs(dE) < 15.0)
            std = np.abs(dE[cidx]).std()
            men = np.mean(dE[cidx])
            print(men, std, men + std)
            idx = np.intersect1d(np.where(dE >= -np.abs(15*std + men))[0], np.where(dE <= np.abs(11*std + men))[0])
            cnt = idx.size
            print('DATADIST: ', dE.size, cnt, (dE.size - cnt), 100.0*((dE.size - cnt)/dE.size))
        E = []
        data_count = np.zeros((N, 3), dtype=np.int32)
        for f in self.h5file:

            adl = pyt.anidataloader(h5d + f)
            for data in adl:
                S = data['species']
                if data[Ekey].size > 0 and (set(S).issubset(self.netdict['atomtyp'])):
                    X = np.array(data['coordinates'], order='C', dtype=np.float32)
                    if Eax0sum:
                        E = energy_unit * np.sum(np.array(data[Ekey], order='C', dtype=np.float64), axis=1)
                    else:
                        E = energy_unit * np.array(data[Ekey], order='C', dtype=np.float64)
                    if forces and not grad:
                        F = forces_unit * np.array(data[Fkey], order='C', dtype=np.float32)
                    elif forces and grad:
                        F = -forces_unit * np.array(data[Fkey], order='C', dtype=np.float32)
                    else:
                        F = 0.0 * X
                    D = np.zeros((E.size,3),dtype=np.float32)
                    if dipole:
                        D = dipole_unit * np.array(data[Dkey], order='C', dtype=np.float32).reshape(E.size,3)
                    else:
                        D = 0.0 * D
                    P = np.zeros((E.size,3,3),dtype=np.float32)
                    if pbc:
                        P = np.array(data['cell'], order='C', dtype=np.float32).reshape(E.size,3,3)
                    else:
                        P = 0.0 * P
                    C = np.zeros((E.size,X.shape[1]),dtype=np.float32)
                    if charge:
                        C = charge_unit * np.array(data[Ckey], order='C', dtype=np.float32).reshape(E.size,len(S))
                    else:
                        C = 0.0 * C
                    if rmhighe:
                        Esae = hdt.compute_sae(self.netdict['saefile'], S)
                        ind_dE = (E - Esae) / np.sqrt(len(S))
                        hidx = np.union1d(np.where(ind_dE < -(15.0 * std + men))[0],
                                          np.where(ind_dE > (11.0 * std + men))[0])
                        lidx = np.intersect1d(np.where(ind_dE >= -(15.0 * std + men))[0],
                                              np.where(ind_dE <= (11.0 * std + men))[0])
                        if hidx.size > 0:
                            print('  -(' + f + ':' + data['path'] + ')High energies detected:\n    ',
                                  (E[hidx] - Esae) / np.sqrt(len(S)))
                        X = X[lidx]
                        E = E[lidx]
                        F = F[lidx]
                        D = D[lidx]
                        C = C[lidx]
                        P = P[lidx]
                    if rmhighf:
                        hfidx = np.where(np.abs(F) > rmhighf)
                        if hfidx[0].size > 0:
                            print('High force:',hfidx)
                            hfidx = np.all(np.abs(F).reshape(E.size,-1) <= rmhighf,axis=1)
                            X = X[hfidx]
                            E = E[hfidx]
                            F = F[hfidx]
                            D = D[hfidx]
                            C = C[hfidx]
                            P = P[hfidx]
                    # Build random split index
                    ridx = np.random.randint(0, Nblocks, size=E.size)
                    Didx = [np.argsort(ridx)[np.where(ridx == i)] for i in range(Nblocks)]
                    # Build training cache
                    for nid, cache in enumerate(cachet):
                        set_idx = np.concatenate(
                            [Didx[((bid + nid * int(Nstride)) % Nblocks)] for bid in range(Ntrain)])
                        if set_idx.size != 0:
                            data_count[nid, 0] += set_idx.size
                            cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx], list(S))
                    for nid, cache in enumerate(cachev):
                        set_idx = np.concatenate(
                            [Didx[(Ntrain + bid + nid * int(Nstride)) % Nblocks] for bid in range(Nvalid)])
                        if set_idx.size != 0:
                            data_count[nid, 1] += set_idx.size
                            cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx], list(S))
                    if build_test:
                        for nid, th5 in enumerate(testh5):
                            set_idx = np.concatenate(
                                [Didx[(Ntrain + Nvalid + bid + nid * int(Nstride)) % Nblocks] for bid in range(Ntest)])
                            if set_idx.size != 0:
                                data_count[nid, 2] += set_idx.size
                                th5.store_data(f + data['path'], coordinates=X[set_idx], forces=F[set_idx], charges=C[set_idx], dipoles=D[set_idx], cell=P[set_idx],energies=E[set_idx], species=list(S))
        # Save train and valid meta file and cleanup testh5
        for t, v in zip(cachet, cachev):
            t.makemetadata()
            v.makemetadata()
        if build_test:
            for th in testh5:
                th.cleanup()



        print(' Train ', ' Valid ', ' Test ')
        print(data_count)
        print('Training set built.')

    def train_ensemble(self, GPUList, remove_existing=False):
        print('Training Ensemble...')
        processes = []
        indicies = np.array_split(np.arange(self.Nn), len(GPUList))
        seeds = np.array_split(np.random.randint(low=0,high=2**32,size=self.Nn), len(GPUList))
        for gpu, (idc,seedl) in enumerate(zip(indicies,seeds)):
            processes.append(Process(target=self.train_network, args=(GPUList[gpu], idc, seedl, remove_existing)))
            processes[-1].start()
        for p in processes:
            p.join()
        print('Training Complete.')

    def train_ensemble_single(self, gpuid, ntwkids, remove_existing=False, random_seed = 0):
        print('Training Single Model From Ensemble...')
        np.random.seed(random_seed)
        random_seeds = np.random.randint(0,2**32,size=len(ntwkids))
        self.train_network(gpuid, ntwkids, random_seeds, remove_existing)
        print('Training Complete.')
    def train_network(self, gpuid, indicies, seeds, remove_existing=False):
        for index,seed in zip(indicies,seeds):
            pyncdict = dict()
            pyncdict['wkdir'] = self.train_root + 'train' + str(index) + '/'
            pyncdict['ntwkStoreDir'] = self.train_root + 'train' + str(index) + '/' + 'networks/'
            pyncdict['datadir'] = self.train_root + "cache-data-" + str(index) + '/'
            pyncdict['gpuid'] = str(gpuid)
            if not os.path.exists(pyncdict['wkdir']):
                os.mkdir(pyncdict['wkdir'])
            if remove_existing:
                shutil.rmtree(pyncdict['ntwkStoreDir'])
            if not os.path.exists(pyncdict['ntwkStoreDir']):
                os.mkdir(pyncdict['ntwkStoreDir'])
            outputfile = pyncdict['wkdir'] + 'output.opt'
            ibuild = copy.deepcopy(self.iptbuilder)
            ibuild.set_parameter('seed',str(seed))
            nfile = pyncdict['wkdir']+'inputtrain.ipt'
            ibuild.write_input_file(nfile,iptsize=self.netdict["iptsize"])
            shutil.copy2(self.netdict['cnstfile'], pyncdict['wkdir'])
            shutil.copy2(self.netdict['saefile'], pyncdict['wkdir'])
            if "/" in nfile:
                nfile = nfile.rsplit("/", 1)[1]
            command = "cd " + pyncdict['wkdir'] + " && HDAtomNNP-Trainer -i " + nfile + " -d " + pyncdict[
                'datadir'] + " -p 1.0 -m -g " + pyncdict['gpuid'] + " > output.opt"
            proc = subprocess.Popen(command, shell=True)
            proc.communicate()
            if 'Termination Criterion Met!' not in open(pyncdict['wkdir']+'output.opt','r').read():
                with open(pyncdict['wkdir']+"output.opt",'a+') as output:
                    output.write("\n!!!TRAINING FAILED TO COMPLETE!!!\n")
            print('  -Model', index, 'complete')
