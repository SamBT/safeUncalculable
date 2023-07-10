import numpy as np
import pythia8
import time
import vector
import fastjet as fj
import awkward as ak
import h5py
import gc

def generate_eeH(jet_type,nEvents,outfile_base,mH=1000,R=0.8,ptMin=20.0):
    print('starting?')
    pythia = pythia8.Pythia()

    # Initialize settings from q/g project
    pythia.readString("PartonShowers:model = 1") # default showers here
    pythia.readString("PDF:lepton = off")
    pythia.readString("PartonLevel:ISR = off")
    pythia.readString("Beams:idA = 11")
    pythia.readString("Beams:idB = -11")
    pythia.readString("Beams:eCM = {0}".format(mH))
    
    pythia.readString("1:m0 = 000")
    pythia.readString("2:m0 = 000")
    pythia.readString("3:m0 = 000")
    pythia.readString("4:m0 = 000")
    pythia.readString("25:m0 = {0}".format(mH))

    pythia.readString("HiggsSM:ffbar2H = on")
    pythia.readString("25:onMode = off")
    
    if jet_type == 'q':
        print("decaying to qqbar (1)")
        pythia.readString("25:onIfAny = 1")
    elif jet_type == 'g':
        print("decaying to gluon (21)")
        pythia.readString("25:onIfAny = 21")
        print("set gluon")
    else:
        print("Invalid jet type!")
        return

    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:Seed = 0")
    print("done settings")

    pythia.init()
    print("pythia initialized")
    
    # generate the events
    nev_gen = 0
    finals = []
    partons = []
    t1 = time.time()
    #pbar = tqdm(range(nEvents))
    while nev_gen < nEvents:
        if not pythia.next(): continue
        nev_gen += 1
        if nev_gen % 1000 == 0:
            print('generated {0} events'.format(nev_gen))
        finals.append([])
        partons.append([])
        for p in pythia.event:
            pdict = {"px":p.px(),"py":p.py(),"pz":p.pz(),"E":p.e()}
            if p.isFinal():
                finals[-1].append(pdict)
            if p.isFinalPartonLevel():
                partons[-1].append(pdict)
        #pbar.update(nev_gen)
        #pbar.refresh()
    finals = ak.Array(finals)
    partons = ak.Array(partons)

    jetdef = fj.JetDefinition(fj.antikt_algorithm,R)
    cluster_f = fj.ClusterSequence(finals,jetdef)
    cluster_p = fj.ClusterSequence(partons,jetdef)

    jets = vector.awk(cluster_f.inclusive_jets(ptMin))
    jet_constits = vector.awk(cluster_f.constituents(min_pt=ptMin))

    pjets = vector.awk(cluster_p.inclusive_jets(ptMin))
    pjet_constits = vector.awk(cluster_p.constituents(min_pt=ptMin))


    # sort by pT, remove any events with 0 jets
    jets_sort = ak.argsort(jets.pt,axis=1,ascending=False)
    jets = jets[jets_sort]
    jet_constits = jet_constits[jets_sort]

    pjet_sort = ak.argsort(pjets.pt,axis=1,ascending=False)
    pjets = pjets[pjet_sort]
    pjet_constits = pjet_constits[pjet_sort]

    mk = (ak.count(pjets.pt,axis=1)>0) & (ak.count(jets.pt,axis=1)>0)
    jets = jets[mk]
    jet_constits = jet_constits[mk]
    pjets = pjets[mk]
    pjet_constits = pjet_constits[mk]
    mk2 = (jets.pt[:,0] >= 400) & (jets.pt[:,0] <= 500)
    jets = jets[mk2]
    jet_constits = jet_constits[mk2]
    pjets = pjets[mk2]
    pjet_constits = pjet_constits[mk2]

    del finals, partons
    gc.collect()

    t2 = time.time()
    print("Took {0:.3f} seconds to generate {1} events".format(t2-t1,nev_gen))
    
    # get leading jet info 
    lead_jets = jets[:,0]
    lead_jet_constits = jet_constits[:,0]
    lead_pjets = pjets[:,0]
    lead_pjet_constits = pjet_constits[:,0]

    dR_mask = lead_jets.deltaR(lead_pjets) < 0.1

    lead_jets = lead_jets[dR_mask]
    lead_jet_constits = lead_jet_constits[dR_mask]

    lead_pjets = lead_pjets[dR_mask]
    lead_pjet_constits = lead_pjet_constits[dR_mask]
    
    deta = lead_jet_constits.deltaeta(lead_jets)
    dphi = lead_jet_constits.deltaphi(lead_jets)
    theta = lead_jet_constits.deltaangle(lead_jets)
    dR = lead_jet_constits.deltaR(lead_jets)
    z = lead_jet_constits.pt / lead_jets.pt
    z_e = lead_jet_constits.E / lead_jets.E

    pdeta = lead_pjet_constits.deltaeta(lead_pjets)
    pdphi = lead_pjet_constits.deltaphi(lead_pjets)
    ptheta = lead_pjet_constits.deltaangle(lead_pjets)
    pdR = lead_pjet_constits.deltaR(lead_pjets)
    pz = lead_pjet_constits.pt / lead_pjets.pt
    pz_e = lead_pjet_constits.E / lead_pjets.E
    
    jmax = 300 # max number of constituents to save
    
    print(f"Asked for {nEvents} events, yielded {len(lead_jets)}")
    
    # write h5
    outDir = "/uscms/home/sbrightt/nobackup/jets-ml/datasets/safeIncalculable/"
    outfile = outDir + outfile_base + "_R{0:.1f}_mH{1}.h5".format(R,mH)
    with h5py.File(outfile,"w") as f:
        f.create_dataset("jet1_pt",data=np.array(lead_jets.pt))
        f.create_dataset("jet1_eta",data=np.array(lead_jets.eta))
        f.create_dataset("jet1_phi",data=np.array(lead_jets.phi))
        f.create_dataset("jet1_e",data=np.array(lead_jets.e))
        f.create_dataset("jet1_constit_z",data=ak.fill_none(ak.pad_none(z,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_ez",data=ak.fill_none(ak.pad_none(z_e,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_pt",data=ak.fill_none(ak.pad_none(lead_jet_constits.pt,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_eta",data=ak.fill_none(ak.pad_none(deta,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_phi",data=ak.fill_none(ak.pad_none(dphi,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_E",data=ak.fill_none(ak.pad_none(lead_jet_constits.E,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_theta",data=ak.fill_none(ak.pad_none(theta,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_dR",data=ak.fill_none(ak.pad_none(dR,jmax,axis=1),0).to_numpy())

        f.create_dataset("pjet1_pt",data=np.array(lead_pjets.pt))
        f.create_dataset("pjet1_eta",data=np.array(lead_pjets.eta))
        f.create_dataset("pjet1_phi",data=np.array(lead_pjets.phi))
        f.create_dataset("pjet1_e",data=np.array(lead_pjets.e))
        f.create_dataset("pjet1_constit_z",data=ak.fill_none(ak.pad_none(pz,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_ez",data=ak.fill_none(ak.pad_none(pz_e,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_pt",data=ak.fill_none(ak.pad_none(lead_pjet_constits.pt,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_eta",data=ak.fill_none(ak.pad_none(pdeta,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_phi",data=ak.fill_none(ak.pad_none(pdphi,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_theta",data=ak.fill_none(ak.pad_none(ptheta,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_dR",data=ak.fill_none(ak.pad_none(pdR,jmax,axis=1),0).to_numpy())
    
    del dR_mask, jets_sort, pjet_sort, mk, mk2
    del jets,lead_jets,z,z_e,deta,dphi,theta,dR
    del pjets,lead_pjets,pz,pz_e,pdeta,pdphi,ptheta,pdR
    del pythia
    gc.collect()

        
def softdrop(jet, zcut=0.1, beta=0, R=0.8):
    parent1, parent2 = fj.PseudoJet(), fj.PseudoJet()
    if not jet.has_parents(parent1, parent2):
        return jet

    pt1, pt2 = parent1.pt(), parent2.pt()
    z = min(pt1, pt2)/(pt1 + pt2)

    if z >= (zcut if beta == 0 else zcut * (parent1.delta_R(parent2)/R)**beta):
        return jet
    else:
        return softdrop(parent1 if pt1 >= pt2 else parent2, zcut=zcut, beta=beta, R=R)
        
def generate_eeH_withSD(jet_type,nEvents,outfile_base,mH=500,R=0.8,ptMin=20.0,dRMatch=0.1):
    pythia = pythia8.Pythia()

    # Initialize settings from q/g project
    pythia.readString("PartonShowers:model = 1") # default showers here
    pythia.readString("PDF:lepton = off")
    pythia.readString("PartonLevel:ISR = off")
    pythia.readString("Beams:idA = 11")
    pythia.readString("Beams:idB = -11")
    pythia.readString("Beams:eCM = {0}".format(mH))
    
    pythia.readString("1:m0 = 000")
    pythia.readString("2:m0 = 000")
    pythia.readString("3:m0 = 000")
    pythia.readString("4:m0 = 000")
    pythia.readString("25:m0 = {0}".format(mH))

    pythia.readString("HiggsSM:ffbar2H = on")
    pythia.readString("25:onMode = off")
    
    if jet_type == 'q':
        pythia.readString("25:onIfAny = 1")
    elif jet_type == 'g':
        pythia.readString("25:onIfAny = 21")
    else:
        print("Invalid jet type!")
        return

    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:Seed = 0")

    pythia.init()
    
    # generate the events
    nev_gen = 0
    t1 = time.time()
    jets = []
    jet_constits = []
    pjets = []
    pjet_constits = []
    jets_sd = []
    jet_sd_constits = []
    pjets_sd = []
    pjet_sd_constits = []
    while nev_gen < nEvents:
        if not pythia.next(): continue
        nev_gen += 1
        if nev_gen % 1000 == 0:
            print('generated {0} events'.format(nev_gen))
        
        finals = []
        partons = []
        
        for p in pythia.event:
            if p.isFinal():
                finals.append(fj.PseudoJet(p.px(),p.py(),p.pz(),p.e()))
            if p.isFinalPartonLevel():
                partons.append(fj.PseudoJet(p.px(),p.py(),p.pz(),p.e()))
        
        jetdef = fj.JetDefinition(fj.antikt_algorithm,R)
        
        cluster_j = fj.ClusterSequence(finals,jetdef)
        js = cluster_j.inclusive_jets(ptMin)
        
        cluster_pj = fj.ClusterSequence(partons,jetdef)
        pjs = cluster_pj.inclusive_jets(ptMin)
        
        if len(js) == 0 or len(pjs) == 0:
            continue
        
        j_pt,pj_pt = 0,0
        j,pj = -1,-1
        
        for jcand in js:
            if jcand.pt() > j_pt:
                j = jcand
                j_pt = jcand.pt()
        for pjcand in pjs:
            if pjcand.pt() > pj_pt:
                pj = pjcand
                pj_pt = pjcand.pt()
        
        if pj.delta_R(j) > dRMatch:
            continue
        jets.append({"px":j.px(),"py":j.py(),"pz":j.pz(),"E":j.e()})
        jet_constits.append([{"px":jc.px(),"py":jc.py(),"pz":jc.pz(),"E":jc.e()} for jc in j.constituents()])
        pjets.append({"px":pj.px(),"py":pj.py(),"pz":pj.pz(),"E":pj.e()})
        pjet_constits.append([{"px":pjc.px(),"py":pjc.py(),"pz":pjc.pz(),"E":pjc.e()} for pjc in pj.constituents()])

        # recluster with Cambridge/Aachen
        jetdef_ca = fj.JetDefinition(fj.cambridge_algorithm,100*R)
        cluster_j_ca = fj.ClusterSequence([jj for jj in j.constituents()],jetdef_ca)
        cluster_pj_ca = fj.ClusterSequence([jj for jj in pj.constituents()],jetdef_ca)
        
        j_ca = cluster_j_ca.inclusive_jets(ptMin)[0]
        pj_ca = cluster_pj_ca.inclusive_jets(ptMin)[0]
        j_sd = softdrop(j_ca, zcut=0.1, beta=0, R=R)
        pj_sd = softdrop(pj_ca, zcut=0.1, beta=0, R=R)

        jets_sd.append({"px":j_sd.px(),"py":j_sd.py(),"pz":j_sd.pz(),"E":j_sd.e()})
        jet_sd_constits.append([{"px":jc.px(),"py":jc.py(),"pz":jc.pz(),"E":jc.e()} for jc in j_sd.constituents()])
        pjets_sd.append({"px":pj_sd.px(),"py":pj_sd.py(),"pz":pj_sd.pz(),"E":pj_sd.e()})
        pjet_sd_constits.append([{"px":pjc.px(),"py":pjc.py(),"pz":pjc.pz(),"E":pjc.e()} for pjc in pj_sd.constituents()])
        
        del jetdef, cluster_j, js, cluster_pj, pjs, j, pj, jetdef_ca, cluster_j_ca, cluster_pj_ca, j_ca, pj_ca, j_sd, pj_sd
            
    jets = vector.awk(jets)
    jets_sd = vector.awk(jets_sd)
    pjets = vector.awk(pjets)
    pjets_sd = vector.awk(pjets_sd)
    
    jet_constits = vector.awk(jet_constits)
    pjet_constits = vector.awk(pjet_constits)
    jet_sd_constits = vector.awk(jet_sd_constits)
    pjet_sd_constits = vector.awk(pjet_sd_constits)

    t2 = time.time()
    print("Took {0:.3f} seconds to generate {1} events".format(t2-t1,nev_gen))
    
    deta = jet_constits.deltaeta(jets)
    dphi = jet_constits.deltaphi(jets)
    theta = jet_constits.deltaangle(jets)
    dR = jet_constits.deltaR(jets)
    z = jet_constits.pt / jets.pt
    jmax = ak.max(ak.count(z,axis=1))
    
    deta_sd = jet_sd_constits.deltaeta(jets_sd)
    dphi_sd = jet_sd_constits.deltaphi(jets_sd)
    theta_sd = jet_sd_constits.deltaangle(jets_sd)
    dR_sd = jet_sd_constits.deltaR(jets_sd)
    z_sd = jet_sd_constits.pt / jets_sd.pt
    jmax_sd = ak.max(ak.count(z_sd,axis=1))

    pdeta = pjet_constits.deltaeta(pjets)
    pdphi = pjet_constits.deltaphi(pjets)
    ptheta = pjet_constits.deltaangle(pjets)
    pdR = pjet_constits.deltaR(pjets)
    pz = pjet_constits.pt / pjets.pt
    pjmax = ak.max(ak.count(pz,axis=1))
    
    pdeta_sd = pjet_sd_constits.deltaeta(pjets_sd)
    pdphi_sd = pjet_sd_constits.deltaphi(pjets_sd)
    ptheta_sd = pjet_sd_constits.deltaangle(pjets_sd)
    pdR_sd = pjet_sd_constits.deltaR(pjets_sd)
    pz_sd = pjet_sd_constits.pt / pjets_sd.pt
    pjmax_sd = ak.max(ak.count(pz_sd,axis=1))
    
    # write h5
    outfile = outfile_base + "_R{0:.1f}_mH{1}_softDrop.h5".format(R,mH)
    with h5py.File(outfile,"w") as f:
        f.create_dataset("jet1_pt",data=np.array(jets.pt))
        f.create_dataset("jet1_eta",data=np.array(jets.eta))
        f.create_dataset("jet1_phi",data=np.array(jets.phi))
        f.create_dataset("jet1_e",data=np.array(jets.e))
        f.create_dataset("jet1_constit_z",data=ak.fill_none(ak.pad_none(z,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_eta",data=ak.fill_none(ak.pad_none(deta,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_phi",data=ak.fill_none(ak.pad_none(dphi,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_theta",data=ak.fill_none(ak.pad_none(theta,jmax,axis=1),0).to_numpy())
        f.create_dataset("jet1_constit_dR",data=ak.fill_none(ak.pad_none(dR,jmax,axis=1),0).to_numpy())
        
        f.create_dataset("jet1_sd_pt",data=np.array(jets_sd.pt))
        f.create_dataset("jet1_sd_eta",data=np.array(jets_sd.eta))
        f.create_dataset("jet1_sd_phi",data=np.array(jets_sd.phi))
        f.create_dataset("jet1_sd_e",data=np.array(jets_sd.e))
        f.create_dataset("jet1_sd_constit_z",data=ak.fill_none(ak.pad_none(z_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("jet1_sd_constit_eta",data=ak.fill_none(ak.pad_none(deta_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("jet1_sd_constit_phi",data=ak.fill_none(ak.pad_none(dphi_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("jet1_sd_constit_theta",data=ak.fill_none(ak.pad_none(theta_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("jet1_sd_constit_dR",data=ak.fill_none(ak.pad_none(dR_sd,jmax_sd,axis=1),0).to_numpy())

        f.create_dataset("pjet1_pt",data=np.array(pjets.pt))
        f.create_dataset("pjet1_eta",data=np.array(pjets.eta))
        f.create_dataset("pjet1_phi",data=np.array(pjets.phi))
        f.create_dataset("pjet1_e",data=np.array(pjets.e))
        f.create_dataset("pjet1_constit_z",data=ak.fill_none(ak.pad_none(pz,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_eta",data=ak.fill_none(ak.pad_none(pdeta,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_phi",data=ak.fill_none(ak.pad_none(pdphi,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_theta",data=ak.fill_none(ak.pad_none(ptheta,jmax,axis=1),0).to_numpy())
        f.create_dataset("pjet1_constit_dR",data=ak.fill_none(ak.pad_none(pdR,jmax,axis=1),0).to_numpy())
        
        f.create_dataset("pjet1_sd_pt",data=np.array(pjets_sd.pt))
        f.create_dataset("pjet1_sd_eta",data=np.array(pjets_sd.eta))
        f.create_dataset("pjet1_sd_phi",data=np.array(pjets_sd.phi))
        f.create_dataset("pjet1_sd_e",data=np.array(pjets_sd.e))
        f.create_dataset("pjet1_sd_constit_z",data=ak.fill_none(ak.pad_none(pz_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("pjet1_sd_constit_eta",data=ak.fill_none(ak.pad_none(pdeta_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("pjet1_sd_constit_phi",data=ak.fill_none(ak.pad_none(pdphi_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("pjet1_sd_constit_theta",data=ak.fill_none(ak.pad_none(ptheta_sd,jmax_sd,axis=1),0).to_numpy())
        f.create_dataset("pjet1_sd_constit_dR",data=ak.fill_none(ak.pad_none(pdR_sd,jmax_sd,axis=1),0).to_numpy())