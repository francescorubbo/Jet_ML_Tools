#ifndef FCNCTOOLS_H
#define FCNCTOOLS_H 

#include <vector>
#include <math.h>
#include <string>

#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"  

#include "Pythia8/Pythia.h"

#include "myFastJetBase.h"

using namespace std;
using fastjet::PseudoJet;

class ReclusterTools {
    private:
        int m_test;



    public:
        ReclusterTools();
        
        // methods
        double JetCharge(fastjet::PseudoJet jet,double kappa);
	bool IsBHadron(int pdgId);
	bool IsCHadron(int pdgId);
	bool Btag(fastjet::PseudoJet jet,vector<fastjet::PseudoJet> bhadrons,vector<fastjet::PseudoJet> chadrons,double jetrad,double b, double c, double uds);
	bool BosonMatch(fastjet::PseudoJet jet, vector<fastjet::PseudoJet> Bosons, double jetrad, int BosonID);
	bool IsIsolated(Pythia8::Particle* particle, Pythia8::Pythia* pythia8, float rel_iso, float conesize);
	int Match(fastjet::PseudoJet jet,vector<fastjet::PseudoJet> jets);
};

#endif

