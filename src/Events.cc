/**
 * Patrick Komiske, Frédéric Dreyer, MIT, 2017
 */

#include "Pythia8/Pythia.h"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "CleverStream.hh"
#include "CmdLine.hh"

using namespace Pythia8;
using namespace std;

#define OUTPRECISION 12
#define MAX_KEPT 1
#define PRINT_FREQ 100

int main(int argc, char** argv) {

  CmdLine cmdline(argc, argv);
  
  // Settings
  int    nEvent    = cmdline.value("-nev", 10000);
  double pthatmin  = cmdline.value("-pthatmin", 25.);
  double pthatmax  = cmdline.value("-pthatmax", -1.);
  double ptpow     = cmdline.value("-ptpow", -1.);
  bool   do_UE     = !cmdline.present("-noUE");
  bool   do_hadr   = !cmdline.present("-parton");
  bool   do_FSR    = !cmdline.present("-noFSR");
  bool   do_ISR    = !cmdline.present("-noISR");
  bool   qcd       = cmdline.present("-allqcd");
  bool   Zg        = cmdline.present("-Zg");
  bool   Zq        = cmdline.present("-Zq");
  double Rparam    = cmdline.value("-R", 0.4);
  double rapMax    = cmdline.value("-rapmax", 2.);
  double ptjetmin  = cmdline.value("-ptjetmin", 50.);
  string filename  = cmdline.value<string>("-out","-");
  int    seed      = cmdline.value("-seed", 0);

  // output setup
  CleverOFStream outstream(filename);
  outstream << "# " << cmdline.command_line() << endl;
  outstream << "# date: " << cmdline.time_stamp() << endl;
   
  cmdline.assert_all_options_used();
  
  // Generator
  Pythia pythia;

  // Specify processes
  assert(qcd || Zg || Zq);
  pythia.settings.flag("HardQCD:all", qcd);
  pythia.settings.flag("WeakBosonAndParton:qqbar2gmZg", Zg);
  pythia.settings.flag("WeakBosonAndParton:qg2gmZq", Zq);

  // Z decay settings
  pythia.readString("WeakZ0:gmZmode = 2");
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfAny = 12 14 16");

  // Random seed
  pythia.settings.flag("Random:setSeed", true);
  pythia.settings.mode("Random:seed", seed);

  // generation cuts and ptpow
  pythia.settings.parm("PhaseSpace:pTHatMin", pthatmin);
  pythia.settings.parm("PhaseSpace:pTHatMax", pthatmax);
  pythia.settings.parm("PhaseSpace:bias2SelectionPow", ptpow);
  pythia.settings.flag("PhaseSpace:bias2Selection", ptpow >= 0 ? true : false);
  
  // Multiparton Interactions, hadronisation, ISR, FSR
  pythia.settings.flag("PartonLevel:MPI", do_UE);
  pythia.settings.flag("PartonLevel:ISR", do_ISR);
  pythia.settings.flag("PartonLevel:FSR", do_FSR);
  pythia.settings.flag("HadronLevel:Hadronize", do_hadr);

  // Initialisation
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = 13000.");

  // Turn off default event listing
  pythia.readString("Next:numberShowEvent = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.init();

  // Jet clustering setup
  fastjet::JetDefinition jet_def = fastjet::JetDefinition(fastjet::antikt_algorithm, Rparam);
  std::vector <fastjet::PseudoJet> particles;
  fastjet::Selector jet_selector = fastjet::SelectorPtMin(ptjetmin) && 
                                   fastjet::SelectorAbsRapMax(rapMax) &&
                                   fastjet::SelectorNHardest(MAX_KEPT);

  outstream << "# Jet algorithm is anti-kT with R=" << Rparam << endl;
  outstream << "# Multiparton interactions are switched "
      << ( (do_UE) ? "on" : "off" ) << endl;
  outstream << "# Hadronisation is "
      << ( (do_UE) ? "on" : "off" ) << endl;
  outstream << "# Final-state radiation is "
      << ( (do_UE) ? "on" : "off" ) << endl;
  outstream << "# Initial-state radiation is "
      << ( (do_UE) ? "on" : "off" ) << endl;
  outstream << "# Random seed is " << seed << endl;
  outstream << setprecision(OUTPRECISION);

  // Begin event loop. Generate event. Skip if error.
  for (int iEvent = 0; iEvent < nEvent;) {
    if (!pythia.next()) continue;

    // Reset Fastjet input
    particles.resize(0);
    
    // Loop over event record to decide what to pass to FastJet
    for (int i = 0; i < pythia.event.size(); ++i) {
      
      // Final state only, no neutrinoutstream
      if (!pythia.event[i].isFinal() || 
          pythia.event[i].idAbs() == 12 || 
          pythia.event[i].idAbs() == 14 ||
          pythia.event[i].idAbs() == 16) continue;

      // Store as input to Fastjet
      fastjet::PseudoJet particle(pythia.event[i].px(),
                                  pythia.event[i].py(), 
                                  pythia.event[i].pz(), 
                                  pythia.event[i].e());
      particle.set_user_index(pythia.event[i].id());
      particles.push_back(particle);
    }

    if (particles.size() == 0) {
      cerr << "Error: event with no final state particles" << endl;
      continue;
    }
    
    // Run Fastjet with selection
    vector<fastjet::PseudoJet> jets = sorted_by_pt(jet_selector(jet_def(particles)));

    // If we've found a jet
    if (jets.size() > 0) {

      iEvent++;
      if (iEvent % PRINT_FREQ == 0) cout << "Generated " << iEvent 
          << " jets so far..." << endl;

      // output particles
      vector<fastjet::PseudoJet> consts = jets[0].constituents();
      outstream << "Event " << iEvent << ", " 
                << jets[0].rap() << "," 
                << jets[0].phi() << ","
                << jets[0].pt()  << endl;

      for (int j = 0; j < consts.size(); j++) 
        outstream << consts[j].rap() << "," << consts[j].phi() << "," 
                  << consts[j].pt()  << "," << consts[j].user_index() << endl;
      outstream << endl;
    }

  // End of event loop.
  }

  // Statistics
  pythia.stat();

  return 0;
}
