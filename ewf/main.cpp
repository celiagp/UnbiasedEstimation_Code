#include "WrightFisher.h"
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/special_functions/beta.hpp> // for beta function
#include <boost/math/special_functions/binomial.hpp> // for function binomial_coefficient
#include <boost/math/special_functions/erf.hpp> // for erf function
#include <boost/timer/timer.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <fstream>
#include <iostream>
#include <libconfig.h++>
#include <random>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char **argv) {
    vector<double> thetaP_in;
    bool non_neut_in;
    double100 sigma_in;
    int SelPolyDeg_in, SelSetup_in;
    double dom_in;
    vector<double> selCoefs_in;

    libconfig::Config cfg;
    try {
      cfg.readFile("config.cfg");
    } catch (libconfig::FileIOException &e) {
      cerr << "FileIOException occurred. Could not read config.cfg!" << endl;
      return (EXIT_FAILURE);
    } catch (libconfig::ParseException &e) {
      cerr << "Parse error at " << e.getFile() << ":" << e.getLine() << "-"
           << e.getError() << endl;
      return (EXIT_FAILURE);
    }

    if (cfg.lookupValue("nonneutral_entry", non_neut_in)) {
      if (non_neut_in) {
        if (!(cfg.lookupValue("sigma_entry", sigma_in))) {
          cerr << "Error in reading population rescaled selection parameter - "
                  "please check input is correct!";
          return (EXIT_FAILURE);
        }

        if (!(cfg.lookupValue("selSetup_entry", SelSetup_in) &&
            (SelSetup_in == 0 || SelSetup_in == 1 || SelSetup_in == 2))) {
          cerr << "Error in reading selection setup indicator function - "
                  "please check input is correct!";
          return (EXIT_FAILURE);
        }

        if (SelSetup_in == 1) {
          if (!((cfg.lookupValue("dominance_entry", dom_in)) && (dom_in >= 0.0) &&
              (dom_in <= 1.0))) {
            cerr << "Error in reading dominance parameter - please check input "
                    "is correct!";
            return (EXIT_FAILURE);
          }
        }

        if (SelSetup_in == 2) {
          if (!(cfg.lookupValue("polyDeg_entry", SelPolyDeg_in) &&
              (SelPolyDeg_in > 0))) {
            cerr << "Error in reading degree of selection polynomial - please "
                    "check input is correct!";
            return (EXIT_FAILURE);
          }
        }
      }
    } else {
      cerr << "Error in reading non-neutral indicator function - please check "
              "input is correct!";
      return (EXIT_FAILURE);
    }

    const libconfig::Setting &root_ = cfg.getRoot();
    if (root_["theta_entries"].getLength() > 2) {
      cerr << "Mutation vector should contain only two entries!";
    }
    for (int i = 0; i < root_["theta_entries"].getLength(); i++) {
      thetaP_in.push_back((double)root_["theta_entries"][i]);
    }
    if (!(thetaP_in[0] > 0.0) && !(thetaP_in[1] > 0.0)) {
      thetaP_in.clear();
    }

    if (SelSetup_in == 2) {
      if (root_["polyCoeffs_entries"].getLength() != SelPolyDeg_in) {
        cerr << "Mismatch in given selection coefficient vector and specified "
                "degree of polynomial!";
        return (EXIT_FAILURE);
      }
      cout << "The inputted selection polynomial has: " << endl;
      for (int i = 0; i < root_["polyCoeffs_entries"].getLength(); i++) {
        selCoefs_in.push_back((double)root_["polyCoeffs_entries"][i]);
      }
    }

    WrightFisher test(thetaP_in, non_neut_in, sigma_in, SelSetup_in, dom_in,
                      SelPolyDeg_in, selCoefs_in);

    Options o;
    boost::random::mt19937 gen;

      libconfig::Config cfgBridge;
      try {
        cfgBridge.readFile("configBridge.cfg");
      } catch (libconfig::FileIOException &e) {
        cerr << "FileIOException occurred. Could not read configBridge.cfg!"
             << endl;
        return (EXIT_FAILURE);
      } catch (libconfig::ParseException &e) {
        cerr << "Parse error at " << e.getFile() << ":" << e.getLine() << "-"
             << e.getError() << endl;
        return (EXIT_FAILURE);
      }

      bool Absorption;
      if (!(cfgBridge.lookupValue("Absorption_entry", Absorption))) {
        cerr << "Error in reading absorption indicator function - please check "
                "input is correct!";
        return (EXIT_FAILURE);
      }

      const libconfig::Setting &root = cfgBridge.getRoot();
      int nBridges = root["nEndpoints"].getLength(),
          xlen = root["bridgePoints_entry"].getLength(),
          tlen = root["bridgeTimes_entry"].getLength();
      int nSamples = root["nSampleTimes_entry"].getLength(),
          slen = root["sampleTimes_entry"].getLength(),
          nlen = root["nSim_entry"].getLength(),
          mlen = root["meshSize_entry"].getLength();
      int nBridgeChecker = -1;

      if (!((xlen == tlen) && ((slen == nlen) or (nlen == 1)))) { // && (tlen == slen)
        cout << "There is a mismatch in the configuration file input!" << endl;

        if (xlen < tlen) {
          cout << "There are more diffusion start points than start times!"
               << endl;
        }
        if (xlen > tlen) {
          cout << "There are more diffusion start times than start points!"
               << endl;
        }
/*        if (slen < tlen) {
          cout << "There are more diffusion start times than sample times!"
               << endl;
        }
        if (slen > tlen) {
          cout << "There are more diffusion sample times than start times!"
               << endl;
        }*/
        if (xlen < slen) {
          cout << "There are more diffusion start points than sample times!"
               << endl;
        }
        if (xlen > slen) {
          cout << "There are more diffusion sample times than start points!"
               << endl;
        }
        if ((xlen < nlen) and (nlen != 1)) {
          cout
              << "There are more number of samples than diffusion start points!"
              << endl;
        }
        if ((xlen > nlen) and (nlen != 1)) {
          cout
              << "There are more diffusion start points than number of samples!"
              << endl;
        }
        if ((slen < nlen) and (nlen != 1)) {
          cout
              << "There are more number of samples than diffusion sample times!"
              << endl;
        }
        if ((slen > nlen) and (nlen != 1)) {
          cout
              << "There are more diffusion sample times than number of samples!"
              << endl;
        }
        if ((tlen < nlen) and (nlen != 1)) {
          cout << "There are more number of samples than diffusion start times!"
               << endl;
        }
        if ((tlen > nlen) and (nlen != 1)) {
          cout << "There are more diffusion start times than number of samples!"
               << endl;
        }
        cout << "Simulation aborted - please fix configDiffusion.cfg as per "
                "the above suggestions."
             << endl;
        return (EXIT_FAILURE);
      }

      if ((nBridges < 1) || (nSamples < 1)) {
        cerr << "No bridge or sampling information provided! Please amend "
                "configBridge.cfg appropriately!";
        return (EXIT_FAILURE);
      } else {
        nBridgeChecker = 0;
        for (int i = 0; i < nBridges; i++) {
          nBridgeChecker += (int)root["nEndpoints"][i];
        }

        if (nBridgeChecker != nBridges + nSamples) {
          cerr << "Mismatch in nEndpoints and nSampleTimes_entry input in "
                  "configBridge.cfg! Please consult config file for info on "
                  "how to set these two quantities up!";
          return (EXIT_FAILURE);
        }
      }

      vector<double100> bridgePoints, bridgeTimes, sampleTimes;
      vector<int> nSim, meshSize, nEndpoints, nSampleTimes;

      for (int i = 0; i < nBridges; i++) {
        nEndpoints.push_back(root["nEndpoints"][i]);
      }

      for (int i = 0; i < nSamples; i++) {
        nSampleTimes.push_back(root["nSampleTimes_entry"][i]);
      }

      for (int i = 0; i < xlen; i++) {
        bridgePoints.push_back(root["bridgePoints_entry"][i]);
        bridgeTimes.push_back(root["bridgeTimes_entry"][i]);
      }
      for (int i = 0; i < slen; i++) {
        sampleTimes.push_back(root["sampleTimes_entry"][i]);
        if (nlen == 1) {
            nSim.push_back(root["nSim_entry"][0]);
        } else {
            nSim.push_back(root["nSim_entry"][i]);
        }
      }

      vector<double100>::iterator indexbP = bridgePoints.begin();
      vector<double100>::iterator indexbT = bridgeTimes.begin();
      vector<int>::iterator indexnS = nSampleTimes.begin();
      vector<double100>::iterator indexsT = sampleTimes.begin();
      int counter = 1;
      for (vector<int>::iterator nEp = nEndpoints.begin();
           nEp != nEndpoints.end(); nEp++) {
        vector<double100> brPt(indexbP, indexbP + (*nEp)),
            brTs(indexbT, indexbT + (*nEp));
        vector<int> nsT(indexnS, indexnS + (*nEp) - 1);
        indexbP += *nEp;
        indexbT += *nEp;
        indexnS += (*nEp) - 1;
        vector<int>::iterator nsti = nsT.begin();
        for (vector<double100>::iterator bP = brPt.begin(), bT = brTs.begin();
             bP != brPt.end() - 1; bP++, bT++) {
          vector<double100> nsTs(indexsT, indexsT + *nsti);
          indexsT += *nsti;
          nsti++;
        }
        counter++;
      }

      indexbP = bridgePoints.begin();
      indexbT = bridgeTimes.begin();
      indexnS = nSampleTimes.begin();
      indexsT = sampleTimes.begin();
      vector<int>::iterator nS = nSim.begin(), nM = meshSize.begin();
      for (vector<int>::iterator nEp = nEndpoints.begin();
           nEp != nEndpoints.end(); nEp++) {
        vector<double100> brPt(indexbP, indexbP + (*nEp)),
            brTs(indexbT, indexbT + (*nEp));
        vector<int> nsT(indexnS, indexnS + (*nEp) - 1);
        indexbP += *nEp;
        indexbT += *nEp;
        indexnS += (*nEp) - 1;
        vector<int>::iterator nsti = nsT.begin();
        for (vector<double100>::iterator bP = brPt.begin(), bT = brTs.begin();
             bP != brPt.end() - 1; bP++, bT++) {
          vector<double100> nsTs(indexsT, indexsT + *nsti);
          indexsT += *nsti;

          time_t t = time(0); // get time now
          struct tm *now = localtime(&t);
          char bufferSaveFile[80];
          strftime(bufferSaveFile, 80, "%Y-%m-%d-%H:%M:%S", now);
          string absnoabs;
          if (Absorption) {
            absnoabs = "Unconditioned";
          } else {
            absnoabs = "Conditioned";
          }
          string saveFilename = "WFbridge.csv";
        for (vector<double100>::iterator sT = nsTs.begin(); sT != nsTs.end();
               sT++) {
            test.BridgeDiffusionRunner(*nS, *bP, *(bP + 1), *bT, *(bT + 1), *sT,
                                       Absorption, saveFilename, o, gen);
            nS++;
          }
          nsti++;
        }
      }

  return 0;
}
