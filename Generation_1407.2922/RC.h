#ifndef RC_H
#define RC_H


#include <vector>
#include <math.h>
#include <string>

#include "TLorentzVector.h"

#include "fastjet/ClusterSequenceActiveAreaExplicitGhosts.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/PseudoJet.hh"  
#include "fastjet/tools/Filter.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/Subtractor.hh"
#include "fastjet/AreaDefinition.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"


#include "fastjet/contrib/VertexJets.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/Njettiness.hh"

#include "RCConfiguration.h"

#include "ReclusterTools.h"

#include "myFastJetBase.h"

using std::vector;
using namespace fastjet;

// worker classes for selectors for vertex
// Selector                                                                                                                                                                                                  
class SelectorWorkerPileupGhostTrack : public fastjet::SelectorWorker {                                                                                    
public:                                                                                                                                                                                                        

	virtual bool pass(const fastjet::PseudoJet & particle) const {                                                                                                        
	// we check that the user_info_ptr is non-zero so as to make                                                                                                        
	// sure that explicit ghosts don't cause the selector to fail                                                                                                               
		return (particle.has_user_info<MyUserInfo>()                                                                                                                          
			&&  fabs(particle.user_info<MyUserInfo>().charge()) > 0) ;                                                                                           
	}                                                                                                                                                                                                                 

	virtual string description() const {return "comes from pileup interaction";}                                                                                  
};                                                                                                                                                                                                                  

 // Selector                                                                                                                                                                                                  
class SelectorWorkerHardScatterGhostTrack : public fastjet::SelectorWorker {                                                                          
public:                                                                                                                                                                                                        

	virtual bool pass(const fastjet::PseudoJet & particle) const {                                                                                                        
   // we check that the user_info_ptr is non-zero so as to make                                                                                                        
   // sure that explicit ghosts don't cause the selector to fail                                                                                                               
		return (particle.has_user_info<MyUserInfo>()                                                                                                                          
			&&  fabs(particle.user_info<MyUserInfo>().charge()) == 0.) ;                                                                                            
	}                                                                                                                                                                                                                 

	virtual string description() const {return "comes from hs interaction";}                                                                                         
};    



// class with user info for clustered objects.
// contains a list of real constituents, which may otherwise be lost
class RCUserInfo : public fastjet::PseudoJet::UserInfoBase{
public:
	RCUserInfo() {nsub = 0;}
	~RCUserInfo() {}

	vector<fastjet::PseudoJet> GetConstituents() const { return constituents; }

	int 						GetNSubjets() const {return nsub;}
	vector<fastjet::PseudoJet> GetSmallRSubs() const { return smallrsubs; }
	void						IncrementSubjets() { nsub++; }

	void						ClearConstituents(){ constituents.clear(); }
	// for vectors of pseudojets: typical fastjet
	void						AddConstituents(vector<fastjet::PseudoJet> _constituents, fastjet::ClusterSequence* theSequence){
		//constituents.clear();
		for(unsigned int iC = 0; iC < _constituents.size(); iC++){
			AddConstituent(_constituents[iC]);
		}
	}

	void 						AddConstituent(fastjet::PseudoJet toAdd){
		//std::cout << toAdd.pt() << " is the added pt " << std::endl;
		if(toAdd.pt() > 0.00000001) 
			constituents.push_back(toAdd);
	}
	void                                            AddSmallRConstituent(fastjet::PseudoJet toAdd){
	  if(toAdd.pt() > 0.00000001)
	    smallrsubs.push_back(toAdd);
        }


private:
	vector<fastjet::PseudoJet> constituents;
	vector<fastjet::PseudoJet> smallrsubs;
//	vector<fastjet::PseudoJet*> constituents;
	int 						nsub;


};

// class which provides "final" information about jets-- not the constituents
// nsub moments and n-subjets functions are here
class RCFinalInfo : public fastjet::PseudoJet::UserInfoBase{
public:
  RCFinalInfo() {tau32 = -1; tau21 = -1; nsub = -1; d12 = -1; d12_RT = -1; d23 = -1; d23_RT = -1;}
	~RCFinalInfo() {}


	float 						GetTau32() const { return tau32; }
	float 						GetTau21() const { return tau21; }

	void						SetTau32(float _tau32) { tau32 = _tau32; }
	void						SetTau21(float _tau21) { tau21 = _tau21; }

	void 						SetNSubjets(int _nsub) { nsub = _nsub; }
	int 						GetNSubjets() const {return nsub;}

	float  Getd12() const {return d12;}
	float  Getd12_RT() const {return d12_RT;}
	float  Getd23() const {return d23;}
	float  Getd23_RT() const {return d23_RT;}

	void Setd12(float _d12) { d12 = _d12;}
	void Setd12_RT(float _d12_RT) { d12_RT =_d12_RT;}
	void Setd23(float _d23) { d23 =_d23;}
	void Setd23_RT(float _d23_RT) { d23_RT =_d23_RT;}

private:

	double    d12;
	double    d23;
	double    d12_RT;
	double    d23_RT;
	int 						nsub;
	float 						tau32;
	float 						tau21;


};



// class RCTestInfo : public fastjet::PseudoJet::UserInfoBase{
// public:
// 	RCTestInfo() { }
// 	~RCTestInfo() { }
// private:
// 	PseudoJet*					self;
// };

//actual reclustering class. contains a configuration object, which you can configure 
//by grabbing it and manipulatin it. contains GetJet, which gives you the jet you want for the
//reclustering project. ~fin
class RC{
private:

	RCConfiguration*			 config;
	ReclusterTools* 			 tool;
	vector<fastjet::PseudoJet>*	 fullParticles;
	vector<fastjet::PseudoJet>*	 hardscatterParticles;
	//vector<fastjet::PseudoJet>	 pileupParticles;

	bool						 manualConfig;

	fastjet::PseudoJet			 reference_jet;
	fastjet::JetMedianBackgroundEstimator estimator;

	fastjet::AreaDefinition 	 active_area;
	// fastjet::AreaDefinition      active_area_explicit_ghosts;

	bool 						 dynamic_reference;

	bool						 debug;

	void 						 CalculateMoments(fastjet::PseudoJet & nsub_jet);
	void						 AddConstituents(fastjet::PseudoJet* theJet, fastjet::ClusterSequence* theSequence);
	void 						 AddConstituents(vector<fastjet::PseudoJet>& theJets,  fastjet::ClusterSequence* theSequence);

	float						 rho;
	void						 CalculateRho();


	fastjet::PseudoJet 			 FloatingCutJet(fastjet::PseudoJet theJet, float fraction);
    vector<fastjet::PseudoJet>   FloatingCut(vector<fastjet::PseudoJet> jets, float fraction);

	//management enum for GetJetVector
	enum JetVectorRunType{
		PILEUP_ON,
		PILEUP_OFF
	};

	vector<fastjet::PseudoJet>   PileupSubtract(vector<fastjet::PseudoJet>& theJets, JetVectorRunType mode);
	vector<fastjet::PseudoJet> 	 GetJetVector(JetVectorRunType mode);


 	// ---------------- Selectors ----------------------                                                                                                                                            
 	// ----------------------------------------------------------------------                                                                                                                    
	fastjet::Selector SelectorPileupGhostTrack() {                                                                                                                                   
 		return new SelectorWorkerPileupGhostTrack();                                                                                                                                
 	}                                                                                                                                                                                                                   
                                                                                                                                                                                                                     
 	fastjet::Selector SelectorHardScatterGhostTrack() {                                                                                                                         
 		return new SelectorWorkerHardScatterGhostTrack();                                                                                                                      
 	}                                                                                                                                                                                                                   
 	// ----------------------------------------------------------------------  




public:

	RC(vector<fastjet::PseudoJet>* _fullParticles, vector<fastjet::PseudoJet>* _hardscatterParticles);
	~RC();

	void					SetConfiguration(RCConfiguration* _config);
	RCConfiguration* 		GetConfiguration();
	fastjet::PseudoJet		GetJet();

	void					SetStaticReference();
	void					SetDebug(bool debugIn);

};

#endif
