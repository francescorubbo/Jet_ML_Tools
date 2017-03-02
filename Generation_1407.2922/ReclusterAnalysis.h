#ifndef  ReclusterAnalysis_H
#define  ReclusterAnalysis_H

#include <vector>
#include <math.h>
#include <string>

#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"  
#include "fastjet/tools/Filter.hh"
#include "fastjet/Selector.hh"

#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TParticle.h"

#include "ReclusterTools.h"
#include "myFastJetBase.h"
#include "Pythia8/Pythia.h"

#include "RC.h"
#include "RCConfiguration.h"

#include "TH2F.h"

using namespace std;
using namespace fastjet;

class ReclusterAnalysis{
    private:
        int  ftest;
        int  fDebug;
        string fOutName;

        TFile *tF;
        TTree *tT;
        ReclusterTools *tool;
	TH2F* detector;

        // Tree Vars ---------------------------------------
        int              fTEventNumber;
	int fTNPV;

	void SetupInt(int & val, TString name);
	void SetupFloat(float & val, TString name);

	vector<RCConfiguration*> configs;
	vector<TString> names;
	vector<float> pts;
	vector<float> ms;
    vector<float> etas;
	vector<float> nsub21s;
	vector<float> nsub32s;
	vector<int>   nsubs;
	vector<float> d12s;
	vector<float> d12_RTs;
	vector<float> d23s;
	vector<float> d23_RTs;

    public:
        ReclusterAnalysis ();
        ~ReclusterAnalysis ();
        
        void Begin();
        void AnalyzeEvent(int iEvt, Pythia8::Pythia *pythia8,  Pythia8::Pythia *pythia_MB, int NPV);
        void End();
        void DeclareBranches();
        void ResetBranches();
        void Debug(int debug){
            fDebug = debug;
        }
        void SetOutName(string outname){
            fOutName = outname;
        }
       
       	void SetupAnalysis(RC & clusterer, RCConfiguration* myconfig, float & pt, float & mass, float & eta, float & nsub21, float & nsub32, int & nsub, float & d12, float & d23, float & d12_RT, float & d23_RT);
       	void DeclareAllAnalyses();
			

};

#endif

