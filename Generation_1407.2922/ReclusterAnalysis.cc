#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <set>


#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TParticle.h"
#include "TDatabasePDG.h"
#include "TMath.h"


#include "ReclusterAnalysis.h"
#include "ReclusterTools.h"

#include "myFastJetBase.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"  
#include "fastjet/tools/Filter.hh"
#include "fastjet/Selector.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/ClusterSequenceActiveAreaExplicitGhosts.hh"

#include "Pythia8/Pythia.h"

using namespace std;

// Constructor 
ReclusterAnalysis::ReclusterAnalysis(){
    if(fDebug) cout << "ReclusterAnalysis::ReclusterAnalysis Start " << endl;
    ftest = 0;
    fDebug = false;
    fOutName = "test.root";
    tool = new ReclusterTools();

    if(fDebug) cout << "ReclusterAnalysis::ReclusterAnalysis End " << endl;
    
}

// Destructor 
ReclusterAnalysis::~ReclusterAnalysis(){
    delete tool;
}

void ReclusterAnalysis::DeclareAllAnalyses(){

  // declare all configs here! just need these three lines per configuration

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeR(fastjet::antikt_algorithm, 1.0);
  names.push_back("AT_Basic");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRGroomed(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, fastjet::SelectorPtFractionMin(0.05));
  names.push_back("AT_Nominal");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRGroomed(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, fastjet::SelectorPtFractionMin(0.03), true);
  names.push_back("AT_Nominal_Corrected_03");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRGroomed(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, fastjet::SelectorPtFractionMin(0.05), true);
  names.push_back("AT_Nominal_Corrected_05");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRGroomed(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, fastjet::SelectorPtFractionMin(0.1), true);
  names.push_back("AT_Nominal_Corrected_10");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRGroomed(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, fastjet::SelectorPtFractionMin(0.2), true);
  names.push_back("AT_Nominal_Corrected_20");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRGroomed(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, fastjet::SelectorPtFractionMin(0.05), true);
  names.push_back("ATAT_Nominal_Corrected_05");

  // //For all of these, fix algo and use R=0.4.  Vary a few other things. 

  double nomfcut = 0.1;

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFixed(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, 20., false, 0.6); //did have a 0.6 on the end
  names.push_back("AT_r4_20_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, nomfcut, false);
  names.push_back("AT_r4_float_mass_noJV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFixed(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, 20., false);
  names.push_back("AT_r4_20_mass_noJV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFixed(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, 20., false, 0.6);
  names.push_back("AT_r3_20_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, nomfcut, false);
  names.push_back("AT_r3_float_mass_noJV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFixed(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, 20., false);
  names.push_back("AT_r3_20_mass_noJV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, nomfcut, true, 0.6);
  names.push_back("AT_r4_float_massless_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, nomfcut, true, 0.6);
  names.push_back("AT_r3_float_massless_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.2, nomfcut, true, 0.6);
  names.push_back("AT_r2_float_massless_JV");

  //Okay, from now on, just vary little r and the algo

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, nomfcut, false, 0.6);
  names.push_back("AT_r4_float_mass_JV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, nomfcut, false, 0.6);
  names.push_back("AT_r3_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.2, nomfcut, false, 0.6);
  names.push_back("AT_r2_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, 0.05, false, 0.6);
  names.push_back("AT_r4_float05_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, 0.05, false, 0.6);
  names.push_back("AT_r3_float05_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.2, 0.05, false, 0.6);
  names.push_back("AT_r2_float05_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, 0.03, false, 0.6);
  names.push_back("AT_r4_float03_mass_JV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, 0.03, false, 0.6);
  names.push_back("AT_r3_float03_mass_JV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.2, 0.03, false, 0.6);
  names.push_back("AT_r2_float03_mass_JV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, 0.01, false, 0.6);
  names.push_back("AT_r4_float01_mass_JV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, 0.01, false, 0.6);
  names.push_back("AT_r3_float01_mass_JV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.2, 0.01, false, 0.6);
  names.push_back("AT_r2_float01_mass_JV");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, 0.1, false, 0.6);
  names.push_back("AT_r4_float10_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, 0.1, false, 0.6);
  names.push_back("AT_r3_float10_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.2, 0.1, false, 0.6);
  names.push_back("AT_r2_float10_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.4, 0.2, false, 0.6);
  names.push_back("AT_r4_float20_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.3, 0.2, false, 0.6);
  names.push_back("AT_r3_float20_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::antikt_algorithm, 0.2, 0.2, false, 0.6);
  names.push_back("AT_r2_float20_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::cambridge_algorithm, 0.4, 0.05, false, 0.6);
  names.push_back("CA_r4_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::cambridge_algorithm, 0.3, 0.05, false, 0.6);
  names.push_back("CA_r3_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::cambridge_algorithm, 0.2, 0.05, false, 0.6);
  names.push_back("CA_r2_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, 0.4, nomfcut, false, 0.6);
  names.push_back("KT_r4_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, 0.3, nomfcut, false, 0.6);
  names.push_back("KT_r3_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, 0.2, nomfcut, false, 0.6);
  names.push_back("KT_r2_float_mass_JV");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRJVF(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, 0.3, 0.05, 0.6);
  names.push_back("AT_r3_jvfgroomed_05_6");
  
  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRJVF(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, 0.3, 0.05, 0.3);
  names.push_back("AT_r3_jvfgroomed_05_3");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRJVF(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, 0.3, 0.10, 0.6);
  names.push_back("AT_r3_jvfgroomed_10_6");

  configs.push_back(new RCConfiguration());
  configs.back()->SetLargeRJVF(fastjet::antikt_algorithm, 1.0, fastjet::kt_algorithm, 0.3, 0.02, 0.6);
  names.push_back("AT_r3_jvfgroomed_02_6");

  // now we setup variables
  for(int iConfig = 0; iConfig < configs.size(); iConfig++){
    pts.push_back(-1.); ms.push_back(-1.); nsub21s.push_back(-1.); nsub32s.push_back(-1.); nsubs.push_back(-1); etas.push_back(-1.); d12s.push_back(-1.);
    d12_RTs.push_back(-1.); d23s.push_back(-1.); d23_RTs.push_back(-1.);
  }
  for(int iConfig = 0; iConfig < configs.size(); iConfig++){
    SetupFloat(pts[iConfig],     names[iConfig]+"_pt");
    SetupFloat(ms[iConfig],      names[iConfig]+"_m");
    SetupFloat(etas[iConfig],    names[iConfig]+"_eta");
    SetupFloat(nsub32s[iConfig], names[iConfig]+"_nsub32");
    SetupFloat(nsub21s[iConfig], names[iConfig]+"_nsub21");
    SetupInt  (nsubs[iConfig],   names[iConfig]+"_nsub");
    SetupFloat(d12s[iConfig], names[iConfig]+"_d12");
    SetupFloat(d12_RTs[iConfig], names[iConfig]+"_d12_RT");
    SetupFloat(d23s[iConfig], names[iConfig]+"_d23");
    SetupFloat(d23_RTs[iConfig], names[iConfig]+"_d23_RT");
  }

}

// Begin method
void ReclusterAnalysis::Begin(){
   // Declare TTree
   tF = new TFile(fOutName.c_str(), "RECREATE");
   tT = new TTree("EventTree", "Event Tree for Recluster");
   
   // max's automated method 
   DeclareAllAnalyses(); 

   // for shit you want to do by hand
   DeclareBranches();
   ResetBranches();
   
   return;
}

// End
void ReclusterAnalysis::End(){
    
    tT->Write();
    tF->Close();
    return;
}

// Analyze
void ReclusterAnalysis::AnalyzeEvent(int ievt, Pythia8::Pythia* pythia8, Pythia8::Pythia* pythia_MB, int NPV){

    if(fDebug) cout << "ReclusterAnalysis::AnalyzeEvent Begin " << endl;

    // -------------------------
    if (!pythia8->next()) return;
    if(fDebug) cout << "ReclusterAnalysis::AnalyzeEvent Event Number " << ievt << endl;

    // reset branches 
    ResetBranches();
    
    // new event-----------------------
    fTEventNumber = ievt;
    std::vector <fastjet::PseudoJet>           particlesForJets;
    std::vector <fastjet::PseudoJet>           particlesForJets_np;

    //Pileup Loop

    fTNPV = NPV;

    for (int iPU = 0; iPU <= NPV; ++iPU) {
      for (int i = 0; i < pythia_MB->event.size(); ++i) {
        if (!pythia_MB->event[i].isFinal()    ) continue;
        if (fabs(pythia_MB->event[i].id())==12) continue;
        if (fabs(pythia_MB->event[i].id())==14) continue;
        if (fabs(pythia_MB->event[i].id())==13) continue;
        if (fabs(pythia_MB->event[i].id())==16) continue;

       //if (pythia_MB->event[i].pT() < 0.5)     continue; 
      PseudoJet p(pythia_MB->event[i].px(), pythia_MB->event[i].py(), pythia_MB->event[i].pz(),pythia_MB->event[i].e() ); 
      p.reset_PtYPhiM(p.pt(), p.rapidity(), p.phi(), 0.); 
      p.set_user_info(new MyUserInfo(pythia_MB->event[i].id(),i,iPU,true)); 
      particlesForJets.push_back(p); 
      }
      if (!pythia_MB->next()) continue;
    }
   
    // Particle loop -----------------------------------------------------------
    for (int ip=0; ip<pythia8->event.size(); ++ip){

      fastjet::PseudoJet p(pythia8->event[ip].px(), pythia8->event[ip].py(), pythia8->event[ip].pz(),pythia8->event[ip].e() ); 
      p.reset_PtYPhiM(p.pt(), p.rapidity(), p.phi(), 0.); 
      p.set_user_info(new MyUserInfo(pythia8->event[ip].id(),ip,0, false)); //0 for the primary vertex. 

	// particles for jets --------------
        if (!pythia8->event[ip].isFinal() )      continue;
        //if (fabs(pythia8->event[ip].id())  ==11) continue;
        if (fabs(pythia8->event[ip].id())  ==12) continue;
        if (fabs(pythia8->event[ip].id())  ==13) continue;
        if (fabs(pythia8->event[ip].id())  ==14) continue;
        if (fabs(pythia8->event[ip].id())  ==16) continue;

	    particlesForJets.push_back(p);
	    particlesForJets_np.push_back(p);

     } // end particle loop -----------------------------------------------

    //Eta requirement?

    RC clusterer(&particlesForJets, &particlesForJets_np);
    if(fDebug) clusterer.SetDebug(true);

    /*
    // optional last argument sets the JVF cut!
    theConfig->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::cambridge_algorithm, 0.3, 0.05, false, 0.25);
    theJet = clusterer.GetJet();
    fTre_AT_CA3_float_p = theJet.pt(); fTre_AT_CA3_float_m=theJet.m();

    // set the second to last argument to true to get massless input jets
    theConfig->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::cambridge_algorithm, 0.2, 0.05, true, 0.25);
    theJet = clusterer.GetJet();

    // this is how you retrieve more info for the jet
    float tau21 = theJet.user_info<RCFinalInfo>().GetTau21();
    float tau32 = theJet.user_info<RCFinalInfo>().GetTau32();
    int   nsub  = theJet.user_info<RCFinalInfo>().GetNSubjets();

    //std::cout << tau21 << " " << tau32 << " " << nsub << " " << theJet.m() << std::endl;

        theConfig->SetReclusteredFloating(fastjet::antikt_algorithm, 1.0, fastjet::cambridge_algorithm, 0.2, 0.05, false, 0.25);
    theJet = clusterer.GetJet();

    // this is how you retrieve more info for the jet
    float tau21_2 = theJet.user_info<RCFinalInfo>().GetTau21();
    float tau32_2 = theJet.user_info<RCFinalInfo>().GetTau32();
    int   nsub_2  = theJet.user_info<RCFinalInfo>().GetNSubjets();

    //std::cout << "second is " << tau21_2 << " " << tau32_2 << " " << nsub_2 << " " << theJet.m() << std::endl;

    */

    // loop over max's stuff
    for(int iConfig = 0; iConfig < configs.size(); iConfig++){
      //cout << "Going over max analysis " << iConfig << endl;
      //cout << "setup " << iConfig << endl;
      SetupAnalysis(clusterer, configs[iConfig], pts[iConfig], ms[iConfig], etas[iConfig], nsub21s[iConfig], nsub32s[iConfig], nsubs[iConfig], d12s[iConfig], d23s[iConfig], d12_RTs[iConfig],d23_RTs[iConfig]);
      clusterer.SetConfiguration(configs[iConfig]);
      //if (iConfig==1){
      //fastjet::PseudoJet  theJet = clusterer.GetJet();
      //std::cout << "jet " << theJet.px() << " " << theJet.py() << " " << theJet.pz() << std::endl;
      //}
    }

    tT->Fill();

    if(fDebug) cout << "ReclusterAnalysis::AnalyzeEvent End " << endl;
    return;
}

// worker function to actually perform an analysis
void ReclusterAnalysis::SetupAnalysis(RC & clusterer, RCConfiguration* myconfig, float & pt, float & mass, float & eta, float & nsub21, float & nsub32, int & nsub,  float & d12, float & d23, float & d12_RT, float & d23_RT){
  clusterer.SetConfiguration(myconfig);
  //cout << myconfig->GetConfigType() << " is the config type !" << endl;
  fastjet::PseudoJet  theJet = clusterer.GetJet();
  
  if(theJet.has_user_info<RCFinalInfo>()){
    nsub21 = theJet.user_info<RCFinalInfo>().GetTau21();
    nsub32 = theJet.user_info<RCFinalInfo>().GetTau32();
    nsub   = theJet.user_info<RCFinalInfo>().GetNSubjets();
    d12 = theJet.user_info<RCFinalInfo>().Getd12();
    d12_RT= theJet.user_info<RCFinalInfo>().Getd12_RT();
    d23= theJet.user_info<RCFinalInfo>().Getd23();
    d23_RT= theJet.user_info<RCFinalInfo>().Getd23_RT();
  } else {
    nsub21 = -1;
    nsub32 = -1;
    nsub   = -1;
    d12=-1;
    d12_RT=-1;
    d23=-1;
    d23_RT=-1;
  }


  pt     = theJet.pt();
  mass   = theJet.m();
  eta    = theJet.eta();
}


// declate branches
void ReclusterAnalysis::DeclareBranches(){
   
   // Event Properties 
   tT->Branch("EventNumber",               &fTEventNumber,            "EventNumber/I");
   tT->Branch("NPV",               &fTNPV,            "NPV/I");

   //tT->GetListOfBranches()->ls();
    
   return;
}

void ReclusterAnalysis::SetupInt(int & val, TString name){
  tT->Branch(name, &val, name+"/I");
}

void ReclusterAnalysis::SetupFloat(float & val, TString name){
  tT->Branch(name, &val, name+"/F");
}

// resets vars
void ReclusterAnalysis::ResetBranches(){
      // reset branches 
      fTEventNumber                 = -999;
      fTNPV = -1;

}
