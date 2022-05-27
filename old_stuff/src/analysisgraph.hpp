#ifndef ANALYSISGRAPH_H
#define ANALYSISGRAPH_H

#include <cstddef>
#include <vector>

class BayesRRmz;

class AnalysisGraph
{
public:
    AnalysisGraph(BayesRRmz *bayes, size_t maxParallel = 12);
    virtual ~AnalysisGraph();

    virtual void exec(unsigned int numInds,
                      unsigned int numSnps,
                      const std::vector<unsigned int> &markerIndices) = 0;

protected:
    BayesRRmz *m_bayes = nullptr;
    size_t m_maxParallel = 12;
};

#endif // ANALYSISGRAPH_H
