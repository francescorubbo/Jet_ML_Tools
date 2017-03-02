#include "RC.h"

RC::RC(vector<fastjet::PseudoJet>* _fullParticles, vector<fastjet::PseudoJet>* _hardscatterParticles):
	fullParticles(_fullParticles), hardscatterParticles(_hardscatterParticles), dynamic_reference(true){
    
    debug 				 = false;

	config				 = new RCConfiguration();
	tool 				 = new ReclusterTools();

	manualConfig		 = false;


	// fullParticles = hardscatterParticles; 
	// fullParticles.insert(fullParticles.end(), pileupParticles.begin(), pileupParticles.end());
	//fullParticles = hardscatterParticles;

	active_area = fastjet::AreaDefinition(fastjet::active_area_explicit_ghosts);

	CalculateRho(); 
}

RC::~RC(){
	if(config && !manualConfig)
		delete config;
	if(tool)
		delete tool;
}

void RC::SetConfiguration(RCConfiguration* _config){
	if(config && !manualConfig){
		manualConfig = true;
		delete config;
	}
	config = _config;
}

void RC::SetDebug(bool debugIn){
	debug = debugIn;
}

void RC::CalculateRho(){

    fastjet::JetDefinition m_jet_def_forrho(fastjet::kt_algorithm,0.4); 
 
	//fastjet::ClusterSequenceArea* csa = new fastjet::ClusterSequenceArea(*fullParticles,m_jet_def_forrho,active_area);
	fastjet::AreaDefinition active_area_forrho = fastjet::AreaDefinition(fastjet::voronoi_area);

	fastjet::ClusterSequenceArea csa(*fullParticles,m_jet_def_forrho,active_area_forrho);
    fastjet::RangeDefinition range(1.5);
    rho = csa.median_pt_per_unit_area(range);

    //std::cout << "debugging rho calculation: " << rho << " " << fullParticles.size() << std::endl;
    
    // for(unsigned int iP = 0; iP < fullParticles.size(); iP++){
    // 	std::cout << fullParticles[iP].pt() << std::endl;
    // }

    if(rho< 0.000001)
    	rho = 0.000001;

    if(debug) cout << " rho is " << rho << endl;

    // estimator = fastjet::JetMedianBackgroundEstimator(range, m_jet_def_forrho, active_area);

    // estimator.set_particles(*fullParticles);


    //delete csa;
}

void RC::SetStaticReference(){
	dynamic_reference = false;

	fastjet::JetDefinition staticDefinition(fastjet::antikt_algorithm, 1.0);
	fastjet::ClusterSequenceArea cs(*hardscatterParticles, staticDefinition, active_area);

	vector<fastjet::PseudoJet> sortedJets = fastjet::sorted_by_pt(cs.inclusive_jets());
	reference_jet = sortedJets[0];

	if(debug) std::cout << "reference jet pt = " << reference_jet.pt() << " " <<reference_jet.eta() << " " << reference_jet.phi() << std::endl;
}

RCConfiguration* RC::GetConfiguration(){
	return config;
}

fastjet::PseudoJet RC::GetJet(){
	if(debug) std::cout << "Inside GetJet -- getting jet with config " << config->GetConfigType() << std::endl;

	//fastjet::contrib::VertexJets vertexjets(0.6);
	
	// here, if necessary, we do everything and get the zeroth jet of the type 
	// when there is no pileup. this becomes the reference_jet, which we compare to

	// if(dynamic_reference){
	// 	if(debug) std::cout << "Setting up dynamic reference jet " << std::endl;
	// 	vector<fastjet::PseudoJet> no_pileup = GetJetVector(PILEUP_OFF);
	// 	reference_jet = no_pileup[0];
	// }

	// now, we get the list of jets ~with~ pileup. This is then matched to
	// the reference_jet, and we get our jet, which we can calculate n-subjettiness for
	// and then return a copy of it at the end.
	if(debug) std::cout << "Getting jets with pileup " << std::endl;
	vector<fastjet::PseudoJet> pileup_jets = GetJetVector(PILEUP_ON);

	if(debug){
		std::cout << " Leading 4 jet pt " ;
		for(unsigned int iJet = 0; iJet < pileup_jets.size() && iJet < 5; iJet++){
			std::cout << pileup_jets[iJet].pt() << " " << pileup_jets[iJet].eta() << " " << pileup_jets[iJet].phi() << " ||| "  ;
		}
		std::cout << std::endl;
	}
	if(debug) std::cout << "fetching the matching jet to the reference" << std::endl;


	// get the match to the output
	//int matched = tool->Match(reference_jet, pileup_jets);

	//if(debug) std::cout << "matched number = " << matched << std::endl;

	fastjet::PseudoJet outputJet;

	// if(matched >= 0)
	// 	outputJet = pileup_jets[matched];
	// else
	// 	outputJet = PseudoJet();



	// currently a hack: just do the leading thing all the time

  if(pileup_jets.size()){
    outputJet = pileup_jets[0];
    if(debug) cout << " outputJet is " << outputJet.pt() << endl;
    if(debug) std::cout << " first pt of consituent is " << outputJet.user_info<RCUserInfo>().GetConstituents()[0].pt() << std::endl;
    CalculateMoments(outputJet);
  }
  else{
    outputJet = PseudoJet(-1,-1,-1,-1);
  }
  if(debug) cout << "returning jet!" << endl;
	return outputJet;
}

// get a vector of jets according to the particles and the configuration of the class.
// also assign all the proper constituents throughout using the RCUserInfo class
// this is a bit wasteful because it's a deep copy of the jets, but probably fine.
// note that the clustersequence is going to get abandoned deep in here. 
// this is fine for the current setup of the project, but if we want more info, it 
// might get annoying. probably easiest to add more to RCUserInfo in that case,
// instead of exposing the actual ClusterSequence to the outside.
// note that this is where we implmement the jvf cut as well.
vector<fastjet::PseudoJet> RC::GetJetVector(JetVectorRunType mode){

	int nPU = 0;
	for(int iC = 0; iC < fullParticles->size(); iC++){
		if(fabs(fullParticles->at(iC).user_info<MyUserInfo>().charge())){
			nPU++;
		}
	}
	fastjet::contrib::VertexJets    vertexjet(SelectorPileupGhostTrack(), SelectorHardScatterGhostTrack());
	vertexjet.set_tot_n_pu_tracks      (nPU);  
	vertexjet.set_ghost_scale_factor   (1);  // kGhostScaleFact = scale factor used to rescale the ghost particles, e.g. 1E-10
	vertexjet.set_corrJVF_scale_factor (0.01);                     // does not matter much, leave it fixed
	vertexjet.set_corrJVF_cut          (-1);                               // no cut is applied but corrJVF is calculated




	fastjet::Selector zeroPT = fastjet::SelectorPtMin(0.0000001);

	if(debug) std::cout << "setting up largeDefinition" << std::endl;
	fastjet::JetDefinition largeDefinition(config->GetLargeAlgorithm(), config->GetLargeR());

	ConfigTypes configType = config->GetConfigType();
	fastjet::ClusterSequence* theLargeSequence;

	if(debug) std::cout << "setting up particle choices" << std::endl;
	vector<fastjet::PseudoJet>* particles;
	if(mode==PILEUP_ON){
		particles = fullParticles;
	} else if (mode==PILEUP_OFF){
		particles = hardscatterParticles;
	}
	vector<fastjet::PseudoJet> ptCutSmallJets;

	if(debug) std::cout << "setting up the clustersequence" << std::endl;
	if(configType==LARGE_R || configType==LARGE_R_GROOMED || configType==LARGE_R_JVF){
		// this is easy: all we do is setup the cluster sequence in the normal way
		// grooming will be handled later
		if(debug) std::cout << "setting up large r jets! " << std::endl;
		theLargeSequence = new fastjet::ClusterSequenceArea(*particles, largeDefinition, active_area);

	} else if (configType==RECLUSTERED_FIXED || configType==RECLUSTERED_FLOATING){

		if(debug) cout << " setting up reclustered jets" << endl;
		fastjet::JetDefinition smallDefinition(config->GetSmallAlgorithm(), config->GetSmallR());
		fastjet::ClusterSequenceArea theSmallSequence(*particles, smallDefinition, active_area);

		vector<fastjet::PseudoJet> smallJetsNoSub = fastjet::sorted_by_pt(theSmallSequence.inclusive_jets());

		if(debug) cout << " have the small_r jets!" << endl;

		vector<fastjet::PseudoJet> smallJetsWithGhosts = PileupSubtract(smallJetsNoSub, mode);
		vector<fastjet::PseudoJet> smallJets = zeroPT(smallJetsWithGhosts);



		fastjet::Selector ptCut;
		if(configType==RECLUSTERED_FIXED){
			ptCut = fastjet::SelectorPtMin(config->GetFixedCut());
		} else if(configType==RECLUSTERED_FLOATING){
			// put the minimum pt here, then do the rest as a dynamic thing like trimming
			ptCut = fastjet::SelectorPtMin(15.);
		}
		ptCutSmallJets = ptCut(smallJets);

		// now apply JVF, if asked for
		
		vector<fastjet::PseudoJet> finalSmallJets;
		float cut = config->GetJVFCut();
		if(cut > -900){
			for(int iJet = 0; iJet < ptCutSmallJets.size(); iJet++){
				fastjet::PseudoJet theJet = vertexjet(ptCutSmallJets[iJet]);
				float jvf = theJet.structure_of<fastjet::contrib::VertexJets>().corrJVF();
				if(jvf > cut){
					finalSmallJets.push_back(theJet);
				}
			}
		} else {
			finalSmallJets = ptCutSmallJets;
		}



			
		// add our constituents so we have them in our own store
		// if we reset in the massless section, the original constituents are LOST	
		AddConstituents(finalSmallJets, &theSmallSequence);


		if(config->GetMassless()){
			for(int iJet = 0; iJet < finalSmallJets.size(); iJet++){
				finalSmallJets[iJet].reset_momentum_PtYPhiM(finalSmallJets[iJet].pt(),
														finalSmallJets[iJet].rapidity(),
														finalSmallJets[iJet].phi(),
														0.);
			}
		}


		//std::cout << ptCutSmallJets[0].has_user_info<RCUserInfo>() << " for before the block " << std::endl;

		theLargeSequence = new fastjet::ClusterSequence(finalSmallJets, largeDefinition);

		vector<fastjet::PseudoJet> test = fastjet::sorted_by_pt(theLargeSequence->inclusive_jets());
		//std::cout << test[0].constituents()[0].has_user_info<RCUserInfo>() << " for inside the block " << std::endl;

	} else if (configType==ITER_RECLUSTERED_FIXED || configType==ITER_RECLUSTERED_FLOATING){
		std::cout << "THIS IS NOT SUPPORTED YET" << configType << std::endl;
	} else {
		std::cout << "NO SUPPORTED CONFIGURATION FOUND FOR " << configType << std::endl;
		exit(0);
	}
    
    //if(configType==RECLUSTERED_FIXED) std::cout << "input has const: " << ptCutSmallJets[0].has_user_info<RCUserInfo>() << std::endl;
	if(debug) std::cout << "getting the output list!" << std::endl;

	vector<fastjet::PseudoJet> theOutputUnsorted = theLargeSequence->inclusive_jets();
	//if(configType==RECLUSTERED_FIXED) std::cout << "did it keep it? " << theOutputUnsorted[0].constituents()[0].has_user_info<RCUserInfo>() << std::endl;

	//if(configType==RECLUSTERED_FIXED) std::cout << "checking right before sorting " << theOutputUnsorted[0].constituents()[0].has_user_info<RCUserInfo>() << std::endl;

	vector<fastjet::PseudoJet> theOutputWithGhosts = fastjet::sorted_by_pt(theOutputUnsorted);
	//if(configType==RECLUSTERED_FIXED) std::cout << "checking right after sorting " << theOutputWithGhosts[0].constituents()[0].has_user_info<RCUserInfo>() << std::endl;

	vector<fastjet::PseudoJet> theOutput = zeroPT(theOutputWithGhosts);


	//if(configType==RECLUSTERED_FIXED) std::cout << ptCutSmallJets[0].has_user_info<RCUserInfo>() << std::endl;
	

	if(debug) std::cout << "checking for grooming" << std::endl;
	// now we implement grooming if necessary
	if(configType==LARGE_R_GROOMED){
		if(debug) std::cout << "doing grooming now!" << std::endl;
		fastjet::Filter filterer;
		//fastjet::Subtractor theSubtractor(rho);
		if(mode==PILEUP_ON){
			filterer = Filter(JetDefinition(config->GetFilteringAlgorithm(), 0.3), config->GetFilteringSelector(), rho);	
			//filterer = Filter(config->GetFilteringAlgorithm(), config->GetFilteringSelector(),rho);	
		} else {
			filterer = Filter(JetDefinition(config->GetFilteringAlgorithm(), 0.3), config->GetFilteringSelector());	
		}
			
		vector<fastjet::PseudoJet> outTemp = theOutput;
		theOutput.clear();
		
		if(config->GetExtraSub()){
			outTemp = PileupSubtract(outTemp, mode);
		}
		
		for(unsigned int iJet = 0; iJet < outTemp.size(); iJet++){
			fastjet::PseudoJet temp = filterer(outTemp[iJet]);
			//std::cout << "whyyyy??? " << temp.area() << std::endl;
			theOutput.push_back(temp);
		}

	} 

	if(configType==LARGE_R_JVF){
			// VertexJet for largeR jets: corrJVFcut 0.6, trimming 0.5                                                                                                                          
		fastjet::contrib::VertexJets    vertexjetLarge(SelectorPileupGhostTrack(), SelectorHardScatterGhostTrack(),fastjet::JetDefinition(config->GetSmallAlgorithm(), config->GetSmallR()));                                                                                                    
		vertexjetLarge.set_tot_n_pu_tracks      (nPU);                                                                                                                        
		vertexjetLarge.set_ghost_scale_factor   (1);                                                                                                                                               
		vertexjetLarge.set_corrJVF_scale_factor (0.01);                                                                                                                                       
		vertexjetLarge.set_corrJVF_cut          (config->GetJVFCut());                                                                                                                                                 
		vertexjetLarge.set_trimming_fcut        (config->GetFloatingCut()); 
		vector<fastjet::PseudoJet> outTemp = theOutput;
		theOutput.clear();
		for(unsigned int iJet = 0; iJet < outTemp.size(); iJet++){
			
			fastjet::PseudoJet temp = vertexjetLarge(outTemp[iJet]);
			theOutput.push_back(temp);
		}
		
	}

	if(debug) std::cout << "checking for extra pileup subtraction" << std::endl;
	// now we implement pileup subtraction for large r jets
	if(configType==LARGE_R){
		if(debug) std::cout << "doing extra pileup subtraction" << std::endl;
		vector<fastjet::PseudoJet> tempOut = theOutput;
		theOutput.clear();
		theOutput = PileupSubtract(tempOut, mode);
		
	}

	if(configType==RECLUSTERED_FLOATING){
		if(debug) std::cout << "doing the floating reclustering hard part" << std::endl;
		vector<fastjet::PseudoJet> theOutputTemp = theOutput;
		theOutput.clear();
		vector<fastjet::PseudoJet> theOutputTemp2 = FloatingCut(theOutputTemp, config->GetFloatingCut());
		theOutput = fastjet::sorted_by_pt(theOutputTemp2);
	}

	if(debug) std::cout << "adding constituents in RCInfo" << std::endl;

	theOutput = fastjet::sorted_by_pt(theOutput);

	// floating has already taken care of this, thank you.
	if(configType!=RECLUSTERED_FLOATING){
		//std::cout << "adding the cons! " << std::endl;
		if(configType==RECLUSTERED_FIXED){ 
			//std::cout << "actual check " << ptCutSmallJets[0].has_user_info<RCUserInfo>() << std::endl;
			//std::cout << "const check " << theOutput[0].constituents()[0].has_user_info<RCUserInfo>() << std::endl;
		}

		AddConstituents(theOutput, theLargeSequence);
	}



	// cleanup
	if(debug) std::cout << "deleting theLargeSequenc" << std::endl;
	delete theLargeSequence;

	if(debug) std::cout << "finished with getting jets" << std::endl;

	return theOutput;
}

// Either do pileup subtraction, or return the input, depending 
// on the settings of the analysis
vector<fastjet::PseudoJet> RC::PileupSubtract(vector<fastjet::PseudoJet>& theJets, JetVectorRunType mode){

	//std::cout << " doing pileup subtraction, with rho = " << rho << std::endl;
	if (mode==PILEUP_ON){

		// fastjet::JetDefinition blahDefinition(fastjet::antikt_algorithm, 0.4);
		// ClusterSequence testSequence(*hardscatterParticles, blahDefinition);

		//vector<fastjet::PseudoJet> testJets = fastjet::sorted_by_pt(testSequence.inclusive_jets()); 
		// std::cout << " pileup subtraction is ON" << std::endl;
		fastjet::Subtractor theSubtractor(rho);
		vector<fastjet::PseudoJet> outputJets;
		for(unsigned int iJet = 0; iJet < theJets.size(); iJet++){
			// cout << "before subtraction " << theJets[iJet].pt() << " " << theJets[iJet].m() << " " << theJets[iJet].e() << " " << theJets[iJet].eta() << " " << theJets[iJet].phi() <<endl;
			outputJets.push_back(theSubtractor.result(theJets[iJet]));
			// cout << "after  subtraction " << outputJets[iJet].pt() << " " << outputJets[iJet].m() << " " << outputJets[iJet].e() << " " << outputJets[iJet].eta() << " " << outputJets[iJet].phi() << endl;


			// cout << " no pileup  " << testJets[iJet].pt() << " " << testJets[iJet].m() << " " << testJets[iJet].e() <<  " " << testJets[iJet].eta() << " " << testJets[iJet].phi() << endl;
			// fastjet::PseudoJet areaVec = theJets[iJet].area_4vector();
			// cout << " area 4 vector " << areaVec.pt() << "  " << areaVec.m() << " " << areaVec.e() << " " << areaVec.eta() <<  " " << areaVec.phi() << endl;

			

			// TLorentzVector lol(0,0,0,0);
			// lol.SetPtEtaPhiE( areaVec.pt(), testJets[iJet].eta(), areaVec.phi(), areaVec.e() );

			// fastjet::PseudoJet areaVec2( lol.Px(), lol.Py(), lol.Pz(), lol.E() );
			// cout << " area 4 vector2 " << areaVec2.pt() << "  " << areaVec2.m() << " " << areaVec2.e() << " " << areaVec2.eta() << endl;



			// fastjet::PseudoJet stupidJet( outputJets[iJet].px(), outputJets[iJet].py(), outputJets[iJet].pz(), outputJets[iJet].e() );
			// //stupidJet.reset_PtYPhiE( outputJets[iJet].pt(), outputJets[iJet].rapidity(), outputJets[iJet].phi(), outputJets[iJet].e() );
			// cout << " stupid jet " << stupidJet.pt () << " " << stupidJet.m() << " " << stupidJet.e() << " " << stupidJet.eta() << endl;
		}
		return outputJets;
	} else{
		return theJets;
	}
}

void RC::AddConstituents(fastjet::PseudoJet* theJet, fastjet::ClusterSequence* theSequence){

	RCUserInfo* jetInfo = new RCUserInfo();

	// if the jet's constituents have a user info, that means *they also have constituents*
	// need to get the consituents of the consituents in that case
	if(theJet->constituents()[0].has_user_info<RCUserInfo>()){
		
		if(debug) cout << "have a user info: adding constituents through this path" << endl;

		for(int iC = 0; iC < theJet->constituents().size(); iC++){
			jetInfo->AddConstituents(theJet->constituents()[iC].user_info<RCUserInfo>().GetConstituents(), theSequence);
			jetInfo->IncrementSubjets();
			jetInfo->AddSmallRConstituent(theJet->constituents()[iC]);
		}
		
	} else{
		// but otherwise, no problem-- just grab the jet's own constituents and add them
		// to our private store

		if(debug) cout << " no user info: adding standard constituents " << endl;

		vector<fastjet::PseudoJet> theConst = theJet->constituents();
		for(int iC = 0; iC < theConst.size(); iC++){
			//cout << theConst[iC].pt() << " " << endl;
			jetInfo->AddConstituent(theConst[iC]);
			//jetInfo->IncrementSubjets(); // why this???
		}

	}

	theJet->set_user_info(jetInfo);	

	if(debug) cout << " finished with constituents " << endl;

}

// just a looper over the above function
void RC::AddConstituents(vector<fastjet::PseudoJet>& theJets,  fastjet::ClusterSequence* theSequence){
	for(unsigned int iJet = 0; iJet < theJets.size(); iJet++){
		if(theJets[iJet].pt() > 0.000000001)
			AddConstituents(&(theJets[iJet]), theSequence);
		if(debug) cout << "added constituents to jet " << iJet << " out of " << theJets.size() << endl;
	}
}

// Implements a floating jet pt cut for iterative reclustering with trimming
fastjet::PseudoJet RC::FloatingCutJet(fastjet::PseudoJet theJet, float fraction){
	float pt = theJet.pt();
	fastjet::PseudoJet outJet(0,0,0,0);
	RCUserInfo* jetInfo = new RCUserInfo();
	outJet.set_user_info(jetInfo);

	vector<fastjet::PseudoJet> constituents = theJet.constituents();


	for(unsigned int iC = 0; iC < constituents.size(); iC++){
		//std::cout << " small jet pt is " << iC << " out of " << constituents.size() << " " << constituents[iC].pt() << std::endl;

		if(constituents[iC].pt() > fraction * pt){
			outJet += constituents[iC];
			// now need to make sure the original constituents are treated properly

			const RCUserInfo& theInfo = constituents[iC].user_info<RCUserInfo>();
			jetInfo->AddConstituents(theInfo.GetConstituents(), 0);
			jetInfo->IncrementSubjets();
			jetInfo->AddSmallRConstituent(constituents[iC]);
		}
	}

	return outJet;
}

// a looper for the above function
vector<fastjet::PseudoJet> RC::FloatingCut(vector<fastjet::PseudoJet> jets, float fraction){
	if(debug) std::cout << "going to clean " << jets.size() << " jets with fraction " << fraction << std::endl;
	vector<fastjet::PseudoJet> cleaned;
	for(unsigned int iJet = 0; iJet < jets.size(); iJet++){
		cleaned.push_back(FloatingCutJet(jets[iJet], fraction));
	}
	return cleaned;
}

// copy all the interesting additional information to a new user info class
// note the horrible hack for nsubjettiness to allow us to use the custom constituents
// this requires a custom compiled nsubjettiness version from fastjet-- will NOT work out of the box
void RC::CalculateMoments(fastjet::PseudoJet & nsub_jet){
	fastjet::contrib::Nsubjettiness NSub(1, fastjet::contrib::Njettiness::onepass_kt_axes, 1., config->GetLargeR());
	float tau1 = NSub._njettinessFinder.getTau(1, nsub_jet.user_info<RCUserInfo>().GetConstituents());
	float tau2 = NSub._njettinessFinder.getTau(2, nsub_jet.user_info<RCUserInfo>().GetConstituents());
	float tau3 = NSub._njettinessFinder.getTau(3, nsub_jet.user_info<RCUserInfo>().GetConstituents());

	if(debug) std::cout << "Nsub values: " << tau3 << " " << tau2 << " " << tau1 << std::endl;

	int nsub = -1;

	ConfigTypes configType = config->GetConfigType();
	RCFinalInfo* finalInfo = new RCFinalInfo();

	if(configType == LARGE_R)
	  nsub = 1;
	else if(configType == LARGE_R_GROOMED || configType == LARGE_R_JVF){
		nsub = nsub_jet.pieces().size();
		fastjet::ClusterSequence kt_cs(nsub_jet.constituents(), fastjet::JetDefinition(fastjet::kt_algorithm, 1.5, fastjet::E_scheme, fastjet::Best));
		std::vector<fastjet::PseudoJet> kt_jets = kt_cs.inclusive_jets();
		fastjet::PseudoJet reclusteredTrimmedFatJet = kt_jets[0];
		float d12 = sqrt(kt_cs.exclusive_dmerge(1));
                float d23 = sqrt(kt_cs.exclusive_dmerge(2));
                finalInfo->Setd23(d23);
                finalInfo->Setd12(d12);
		//std::cout << " " << " CCC " << d12 << " " << nsub_jet.constituents().size() << std::endl;
	}
	else if(configType == RECLUSTERED_FIXED || configType == RECLUSTERED_FLOATING){
		nsub = nsub_jet.user_info<RCUserInfo>().GetNSubjets();
		//std::cout << "ben check " << nsub << " " << nsub_jet.user_info<RCUserInfo>().GetSmallRSubs().size() << std::endl;
		fastjet::ClusterSequence kt_cs_RT(nsub_jet.user_info<RCUserInfo>().GetSmallRSubs(), fastjet::JetDefinition(fastjet::kt_algorithm, 1.5, fastjet::E_scheme, fastjet::Best));
		std::vector<fastjet::PseudoJet> kt_jets_RT = kt_cs_RT.inclusive_jets();
		fastjet::PseudoJet reclusteredTrimmedFatJet_RT = kt_jets_RT[0];
		fastjet::ClusterSequence kt_cs(nsub_jet.user_info<RCUserInfo>().GetConstituents(), fastjet::JetDefinition(fastjet::kt_algorithm, 1.5, fastjet::E_scheme, fastjet::Best));
		std::vector<fastjet::PseudoJet> kt_jets = kt_cs.inclusive_jets();
		fastjet::PseudoJet reclusteredTrimmedFatJet = kt_jets[0];
		float d12 = sqrt(kt_cs.exclusive_dmerge(1));
		float d23 = sqrt(kt_cs.exclusive_dmerge(2));
		float d12_RT = sqrt(kt_cs_RT.exclusive_dmerge(1));
                float d23_RT = sqrt(kt_cs_RT.exclusive_dmerge(2));
		finalInfo->Setd23_RT(d23_RT);
		finalInfo->Setd12_RT(d12_RT);
		finalInfo->Setd23(d23);
		finalInfo->Setd12(d12);
		//std::cout << "ben check " << nsub << " " << d12 << " " << d12_RT << std::endl;
	}

	finalInfo->SetTau21(tau2 / tau1);
	finalInfo->SetTau32(tau3 / tau2);

	finalInfo->SetNSubjets(nsub);

	nsub_jet.set_user_info(finalInfo);
	if(debug) cout << " finished moments " << endl;
}


