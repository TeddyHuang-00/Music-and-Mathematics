#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <vector>

using namespace std;

// ------------- hyper-parameters -------------- //

// // Music parameters
// constexpr int PitchRange = 8 * 12 + 1;
constexpr int PitchRangeLow = 3 * 12;
constexpr int PitchRangeHigh = 3 * 12 + 1 + PitchRangeLow;
constexpr int DurationRange = 8;
constexpr int BeatsPerBar = 4;
constexpr int DurationPerBeat = 4;

// // Target parameters
constexpr int MediumFrequencyScale = 4;
constexpr int RhythmVariation = 2;
constexpr int PitchVariation = 6;
constexpr double SameDegreeProportion = 0.4;
constexpr int PitchJumpDistance = 6;
constexpr int PitchSpan = 12;

// // Fitness parameters
constexpr int Weight_ChromaticNumber = 50;
constexpr int Weight_ConsecutiveJumps = 50;
constexpr int Weight_HalfNote = 100;
constexpr int Weight_MediumFrequencyScale = 500;
constexpr int Weight_OutOfScale = 25;
constexpr int Weight_HighFrequencyComponent = 5;
constexpr int Weight_PitchDevition = 5;
constexpr int Weight_PitchDiversity = 100;
constexpr int Weight_PitchJump = 75;
constexpr int Weight_PitchSpan = 500;
constexpr int Weight_RhythmDiversity = 200;
constexpr int Weight_RhythmSimilarity = 10;
constexpr int Weight_RhythmVariationByBar = 200;
constexpr int Weight_ScaleConsistency = 50;
constexpr int Weight_StableConsonance = 15;
constexpr int Weight_StepWise = 500;

// // Genetic Algorithm parameters

// Number of notes in each agent
constexpr int N = 60;
// Number of generations
constexpr int M = 100;
// Number of agents
constexpr int K = 1000;
// Keep the best P agents
constexpr int P = 1;
// Mutation rate per mille
constexpr int MutationRate = 50;
// Cross-over rate per mille
constexpr int CrossOverRate = 50;

// // Beta features
// Variable evolution rate
constexpr bool EnableAdaptiveEvolutionRate = true;
constexpr int beta_GenerationInterval = 10;
int beta_MutationRate = MutationRate;
int beta_CrossOverRate = CrossOverRate;
constexpr double beta_MaxMultiplier = 10.0;
// Reference mutation
constexpr bool EnableReferenceMutation = true;
constexpr int beta_PitchMutationRange = 6;
constexpr int beta_DurationMutationRange = 2;
// Make best agent mate with the worst, second mate second, so on
constexpr bool EnableBestMateWorst = false;

// // Output parameters
constexpr int LineBreak = 10;
constexpr int ConsolePrintInterval = 10;
constexpr bool IncludeTimeDate = true;
constexpr bool TrackAllFitness = true;
char OutputPath[] = "music.txt";
char FitnessDataPath[] = "fitness_data.csv";

// -------------------------------------------- //

// ------------------ utils ------------------- //

struct Node;
class Fitness;
typedef pair<Node, Node> par;
typedef complex<double> Complex;

string _date = __DATE__;
string _time = __TIME__;

constexpr int DurationPerBar = BeatsPerBar * DurationPerBeat;
string scaleName[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
auto scale2pitch = map<string, set<int>>{
    {"C", {0, 2, 4, 5, 7, 9, 11}},
    {"C#", {0, 1, 3, 5, 6, 8, 10}},
    {"D", {1, 2, 4, 6, 7, 9, 11}},
    {"D#", {0, 2, 3, 5, 7, 8, 10}},
    {"E", {1, 3, 4, 6, 8, 9, 11}},
    {"F", {0, 2, 4, 5, 7, 9, 10}},
    {"F#", {1, 3, 5, 6, 8, 10, 11}},
    {"G", {0, 2, 4, 6, 7, 9, 11}},
    {"G#", {0, 1, 3, 5, 7, 8, 10}},
    {"A", {1, 2, 4, 6, 8, 9, 11}},
    {"A#", {0, 2, 3, 5, 7, 9, 10}},
    {"B", {1, 3, 4, 6, 8, 10, 11}},
};
auto pitch2scale = map<int, set<string>>{
    {0, {"C", "C#", "D#", "F", "G", "G#", "A#"}},
    {1, {"C#", "D", "E", "F#", "G#", "A", "B"}},
    {2, {"C", "D", "D#", "F", "G", "A", "A#"}},
    {3, {"C#", "D#", "E", "F#", "G#", "A#", "B"}},
    {4, {"C", "D", "E", "F", "G", "A", "B"}},
    {5, {"C", "C#", "D#", "F", "F#", "G#", "A#"}},
    {6, {"C#", "D", "E", "F#", "G", "A", "B"}},
    {7, {"C", "D", "D#", "F", "G", "G#", "A#"}},
    {8, {"C#", "D#", "E", "F#", "G#", "A", "B"}},
    {9, {"C", "D", "E", "F", "G", "A", "A#"}},
    {10, {"C#", "D#", "F", "F#", "G#", "A#", "B"}},
    {11, {"C", "D", "E", "F#", "G", "A", "B"}},
};

// The preference for perfect octave is put here but will not be used in the program
//                               1, 2m,2M,3m,3M, 4, 5-, 5, 6m,6M,7m, 7M, 8
auto consonance = array<int, 13>{3, -2, 2, 3, 3, 4, -2, 4, -1, 2, 2, -2, 3};
auto scaleProbTemplate = map<string, double>{
    {"C", 0},
    {"C#", 0},
    {"D", 0},
    {"D#", 0},
    {"E", 0},
    {"F", 0},
    {"F#", 0},
    {"G", 0},
    {"G#", 0},
    {"A", 0},
    {"A#", 0},
    {"B", 0},
    {"B#", 0},
};

template <typename Container>
typename Container::value_type
auto_correlation(const Container &arr, const int &shift)
{
    typename Container::value_type sum = 0;
    for (int i = shift; i < N; i++)
        sum += arr[i] * arr[i - shift];
    return sum;
}

template <typename Container>
typename Container::value_type
sum(const Container &arr)
{
    typename Container::value_type sum = 0;
    for (auto &i : arr)
        sum += i;
    return sum;
}

template <typename Container>
typename Container::value_type
max(const Container &arr)
{
    typename Container::value_type max = arr[0];
    for (auto &i : arr)
        max = i > max ? i : max;
    return max;
}

template <typename Container>
typename Container::value_type
min(const Container &arr)
{
    typename Container::value_type min = arr[0];
    for (auto &i : arr)
        min = i < min ? i : min;
    return min;
}

template <typename Container>
double mean(const Container &arr)
{
    return 1.0 * sum(arr) / N;
}

template <typename Container>
double stddev(const Container &arr)
{
    double sum = 0;
    double mean_ = mean(arr);
    for (auto &i : arr)
        sum += (i - mean_) * (i - mean_);
    return sqrt(sum / N);
}

template <typename Container>
vector<Complex> FFT(const Container &arr, bool invert)
{
    int bit = 0;
    int n = arr.size();
    while ((1 << bit) < n)
        bit++;
    auto a = vector<Complex>(1 << bit, 0);
    for (int i = 0; i < n; ++i)
        a[i] = (Complex)arr[i];
    n = a.size();
    auto rev = vector<int>(n, 0);
    for (int i = 0; i < n; ++i)
    {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
        if (rev[i] > i)
            swap(a[i], a[rev[i]]);
    }
    for (int len = 2; len <= n; len <<= 1)
    {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        Complex w_0(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len)
        {
            Complex w(1);
            for (int j = 0; j < len / 2; ++j)
            {
                Complex u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= w_0;
            }
        }
    }
    if (invert)
        for (auto &x : a)
            x /= n;
    return a;
}

int hash_array(const array<int, N> &a)
{
    int hash = 0;
    for (auto &i : a)
        hash = (hash + (324723947 + i)) ^ 93485734985;
    return hash;
}

string pitch2name(int pitch)
{
    int scalePartition = pitch / 12;
    int scaleIndex = pitch % 12;
    return scaleName[scaleIndex] + to_string(scalePartition);
}

void beta_AdjustEvolutionRate(const int &num_iters)
{
    beta_MutationRate = MutationRate * min(1.0 + 1.0 * num_iters / beta_GenerationInterval, beta_MaxMultiplier);
    beta_CrossOverRate = CrossOverRate * min(1.0 + 1.0 * num_iters / beta_GenerationInterval, beta_MaxMultiplier);
}

struct Node
{
    // Pitch
    array<int, N> pval;
    // Duration
    array<int, N> tval;
    // Fitness
    int fval;
    double fexp;
    // Unique id
    int hash_id;
    Node() {}

    void init();
    void mutate();
    void eval();
    int id() const;
    int value() const;
    double exp_value() const;
    par makeChild(const Node &x) const;
    void printPitch() const;
    void printName() const;
};

void Node::init()
{
    // Randomly generate the pitch and duration
    for (int i = 0; i < N; i++)
        this->pval[i] = rand() % (PitchRangeHigh - PitchRangeLow) + PitchRangeLow;
    for (int i = 0; i < N; i++)
        this->tval[i] = rand() % DurationRange + 1;
}

void Node::mutate()
{
    // Site-specific mutation
    for (int i = 0; i < N; i++)
    {
        if (rand() % 1000 <= (EnableAdaptiveEvolutionRate ? beta_MutationRate : MutationRate))
        {
            if (EnableReferenceMutation)
            {
                int ref = pval[i] +
                          max(beta_PitchMutationRange / 2 - (pval[i] - PitchRangeLow), 0) -
                          max(beta_PitchMutationRange / 2 - (PitchRangeHigh - pval[i]), 0);
                pval[i] = ref + (rand() % beta_PitchMutationRange) - beta_PitchMutationRange / 2;
            }
            else
                pval[i] = rand() % (PitchRangeHigh - PitchRangeLow) + PitchRangeLow;
        }

        if (rand() % 1000 <= (EnableAdaptiveEvolutionRate ? beta_MutationRate : MutationRate))
        {
            if (EnableReferenceMutation)
            {
                int ref = tval[i] +
                          max(beta_DurationMutationRange / 2 - (tval[i] - 1), 0) -
                          max(beta_DurationMutationRange - (DurationRange - tval[i]), 0);
                tval[i] = ref + (rand() % beta_DurationMutationRange) - beta_DurationMutationRange / 2;
            }
            else
                tval[i] = rand() % DurationRange + 1;
        }
    }
}

int Node::id() const
{
    return this->hash_id;
}

int Node::value() const
{
    return this->fval;
}

double Node::exp_value() const
{
    return this->fexp;
}

par Node::makeChild(const Node &x) const
{
    par res;
    bool crossover_point = false;
    for (int i = 0; i < N; i++)
    {
        if (rand() % 1000 <= (EnableAdaptiveEvolutionRate ? beta_CrossOverRate : CrossOverRate))
            crossover_point = !crossover_point;
        if (!crossover_point)
        {
            res.first.pval[i] = this->pval[i];
            res.first.tval[i] = this->tval[i];
            res.second.pval[i] = x.pval[i];
            res.second.tval[i] = x.tval[i];
        }
        else
        {
            res.first.pval[i] = x.pval[i];
            res.first.tval[i] = x.tval[i];
            res.second.pval[i] = this->pval[i];
            res.second.tval[i] = this->tval[i];
        }
    }
    res.first.mutate();
    res.second.mutate();
    res.first.eval();
    res.second.eval();

    return res;
}

void Node::printPitch() const
{
    for (int i = 0; i < N; i++)
    {
        cout << "(" << setw(3) << this->pval[i] << "," << this->tval[i] << ")";
        if ((i + 1) % LineBreak == 0)
            cout << endl;
        else
            cout << " ";
    }
    cout << endl;
}

void Node::printName() const
{
    for (int i = 0; i < N; i++)
    {
        cout << "(" << setw(3) << pitch2name(this->pval[i]) << "," << this->tval[i] << ")";
        if ((i + 1) % LineBreak == 0)
            cout << endl;
        else
            cout << " ";
    }
    cout << endl;
}

class Fitness
{
public:
    Fitness(const Node *n);
    int eval();

private:
    const Node *node;
    vector<vector<pair<int, int>>> bars;
    vector<pair<string, double>> scaleProb;
    int _ChromaticNumber() const;
    int _ConsecutiveJumps() const;
    int _HalfNote() const;
    int _MediumFrequencyScale() const;
    int _OutOfScale() const;
    int _HighFrequencyComponent() const;
    int _PitchDevition() const;
    int _PitchDiversity() const;
    int _PitchJump() const;
    int _PitchSpan() const;
    int _RhythmDiversity() const;
    int _RhythmSimilarity() const;
    int _RhythmVariationByBar() const;
    int _ScaleConsistency() const;
    int _StableConsonance() const;
    int _StepWise() const;
    // TODO: add more fitness functions
};

Fitness::Fitness(const Node *n) : node(n)
{
    auto bar = vector<pair<int, int>>();
    int t_sum = 0;
    for (auto i = 0; i < this->node->tval.size(); ++i)
    {
        bar.push_back(make_pair(this->node->pval[i], this->node->tval[i]));
        t_sum += this->node->tval[i];
        if (t_sum >= DurationPerBar)
        {
            bars.push_back(bar);
            bar = vector<pair<int, int>>();
            t_sum = 0;
        }
    }
    auto scale_prob = scaleProbTemplate;
    auto counter = array<int, 12>();
    for (auto &p : this->node->pval)
        ++counter[p % 12];
    for (int v = 0; v < 12; ++v)
        for (auto &p : pitch2scale[v])
            scale_prob[p] += 1.0 / this->node->pval.size() / 7 * counter[v];
    copy(scale_prob.begin(), scale_prob.end(), back_inserter(this->scaleProb));
    sort(this->scaleProb.begin(), this->scaleProb.end(),
         [](const pair<string, double> &a, const pair<string, double> &b)
         { return a.second > b.second; });
}

int Fitness::eval()
{
    int fitness = 0;
    // Comment out some of the fitness functions to test the results
    fitness += _ChromaticNumber() * Weight_ChromaticNumber;
    fitness += _ConsecutiveJumps() * Weight_ConsecutiveJumps;
    fitness += _HalfNote() * Weight_HalfNote;
    fitness += _MediumFrequencyScale() * Weight_MediumFrequencyScale;
    fitness += _OutOfScale() * Weight_OutOfScale;
    fitness += _HighFrequencyComponent() * Weight_HighFrequencyComponent;
    fitness += _PitchDevition() * Weight_PitchDevition;
    fitness += _PitchDiversity() * Weight_PitchDiversity;
    fitness += _PitchJump() * Weight_PitchJump;
    fitness += _PitchSpan() * Weight_PitchSpan;
    fitness += _RhythmDiversity() * Weight_RhythmDiversity;
    fitness += _RhythmSimilarity() * Weight_RhythmSimilarity;
    fitness += _RhythmVariationByBar() * Weight_RhythmVariationByBar;
    fitness += _ScaleConsistency() * Weight_ScaleConsistency;
    fitness += _StableConsonance() * Weight_StableConsonance;
    fitness += _StepWise() * Weight_StepWise;
    return fitness;
}

int Fitness::_ChromaticNumber() const
{
    int count = 0;
    int last_p = this->node->pval[0];
    for (auto &p : this->node->pval)
    {
        if (abs(p - last_p) == 1)
            ++count;
        last_p = p;
    }
    return -count;
}

int Fitness::_ConsecutiveJumps() const
{
    int count = 0;
    int last_p = this->node->pval[0];
    bool flag = false;
    for (auto &p : this->node->pval)
    {
        if (abs(p - last_p) > PitchJumpDistance)
        {
            count += flag;
            flag = true;
        }
        else
            flag = false;
        last_p = p;
    }
    return -count;
}

int Fitness::_HalfNote() const
{
    int count = 0;
    int temp_sum = 0;
    for (auto &bar : this->bars)
    {
        for (auto &n : bar)
            temp_sum += n.second;
        // Strictly greater means note crosses bars
        count += temp_sum - DurationPerBar;
        temp_sum = 0;
    }
    // Also counts if the last bar is incomplete
    return -count - (temp_sum > 0);
}

int Fitness::_MediumFrequencyScale() const
{
    int avg_scale = (int)(mean(this->node->pval) / 12);
    return avg_scale == MediumFrequencyScale;
}

int Fitness::_OutOfScale() const
{
    int count = 0;
    auto scale = this->scaleProb.begin();
    for (auto s = this->scaleProb.begin(); s != this->scaleProb.end(); ++s)
        if (s->second > scale->second)
            scale = s;
    auto pitches = scale2pitch[scale->first];
    for (auto &bar : this->bars)
        for (auto &note : bar)
            count += !pitches.contains(note.first % 12);
    return -count;
}

int Fitness::_HighFrequencyComponent() const
{
    // High Frequency is defined as length shorter than 1/4 of the sequence
    double highFreqAmpSum = 0;
    auto normalized = vector<double>(this->node->pval.size());
    auto p_mean = mean(this->node->pval);
    for (int i = 0; i < this->node->pval.size(); ++i)
        normalized[i] = (this->node->pval[i] - p_mean) / 12;
    auto pitch_fft = FFT(normalized, false);
    for (int i = pitch_fft.size() * 3 / 4; i < pitch_fft.size(); ++i)
        highFreqAmpSum += abs(pitch_fft[i]) * abs(pitch_fft[i]);
    return -highFreqAmpSum;
}

int Fitness::_PitchDevition() const
{
    auto p_mean = mean(this->node->pval);
    auto pval = array<double, N>();
    for (int i = 0; i < N; ++i)
        pval[i] = this->node->pval[i] - p_mean;
    return -stddev(this->node->pval);
}

int Fitness::_PitchDiversity() const
{
    auto pitchTypes = set<int>(
        this->node->pval.begin(),
        this->node->pval.end());
    return pitchTypes.size() > PitchVariation;
}

int Fitness::_PitchJump() const
{
    int count = 0;
    int last_p = this->node->pval[0];
    for (auto &p : this->node->pval)
    {
        if (abs(p - last_p) > PitchJumpDistance)
            ++count;
        last_p = p;
    }
    return -count;
}

int Fitness::_PitchSpan() const
{
    return max(this->node->pval) -
               min(this->node->pval) >
           PitchSpan;
}

int Fitness::_RhythmDiversity() const
{
    auto rhythmTypes = set<int>(
        this->node->tval.begin(),
        this->node->tval.end());
    return rhythmTypes.size() > RhythmVariation;
}

int Fitness::_RhythmSimilarity() const
{
    double sum = 0;
    double t_mean = mean(this->node->tval);
    double t_std = stddev(this->node->tval);
    auto tval = array<double, N>();
    for (int i = 0; i < N; ++i)
        tval[i] = (this->node->tval[i] - t_mean) / t_std;
    for (int i = 1; i < N; ++i)
        sum += auto_correlation(tval, i);
    return sum;
}

int Fitness::_RhythmVariationByBar() const
{
    string bar_type = "";
    auto count = set<string>();
    for (auto &bar : this->bars)
    {
        for (auto &n : bar)
            bar_type += to_string(n.second);
        count.insert(bar_type);
        bar_type = "";
    }
    return count.size();
}

int Fitness::_ScaleConsistency() const
{
    auto counter = array<int, 12>();
    double count = 0;
    for (auto &p : this->node->pval)
        ++counter[p % 12];
    for (int v = 0; v < 12; ++v)
        for (auto &s : this->scaleProb)
            if (scale2pitch[s.first].contains(v))
                count += s.second * counter[v];
    return count;
}

int Fitness::_StableConsonance() const
{
    int score = -consonance[0];
    int last_p = this->node->pval[0];
    for (auto &p : this->node->pval)
    {
        score += consonance[abs(p - last_p) % 12];
        last_p = p;
    }
    return score;
}

int Fitness::_StepWise() const
{
    int count = -1;
    int last_p = this->node->pval[0];
    for (auto &p : this->node->pval)
    {
        if (abs(p - last_p) <= 1)
            ++count;
        last_p = p;
    }
    return count < SameDegreeProportion * N;
}

void Node::eval()
{
    this->fval = Fitness(this).eval();
    this->fexp = exp(this->fval / 1000.0);
    this->hash_id = hash_array(this->pval) + hash_array(this->tval);
}

struct NodeCmp
{
    bool operator()(const Node &lhs, const Node &rhs) const
    {
        if (lhs.fval == rhs.fval)
            return lhs.hash_id > rhs.hash_id;
        else
            return lhs.value() > rhs.value();
    }
};

template <typename Container>
Node getNode(const Container &live)
{
    // Select a random node from the live set
    // Probability of selection is related to fitness
    double tot = 0;
    for (auto &x : live)
        tot += x.exp_value();
    double idx = rand() / (RAND_MAX + 1.0) * tot;

    for (auto &x : live)
        if (idx <= x.exp_value())
            return x;
        else
            idx -= x.exp_value();
    return *live.rbegin();
}

template <typename Container>
par beta_getParent(const Container &live, const int idx)
{
    auto left = live.begin();
    auto right = live.rbegin();
    advance(left, idx);
    advance(right, idx);
    return make_pair(*left, *right);
}

template <typename Container>
void sortNodes(Container &live)
{
    sort(live.begin(), live.end(), NodeCmp());
}

// -------------------------------------------- //

vector<Node> parent, children;

int main()
{
    srand(time(0));
    auto all_fitness = vector<vector<int>>();
    int BestFitness = 0;
    int BestFitnessCounter = 0;
    // Init
    for (int i = 0; i < K; i++)
    {
        Node n;
        n.init();
        n.eval();
        children.push_back(n);
    }
    sortNodes(children);
    // M Generations
    for (int T = 0; T <= M; T++)
    {
        if (T % ConsolePrintInterval == 0)
            cout << "Generation" << setw(6) << T << '\t'
                 << "Fitness" << setw(7) << children.begin()->value() << endl;

        parent = children;
        children.clear();

        // Keep the best P agents
        if (P)
        {
            auto pos = parent.begin();
            advance(pos, P);
            children.insert(children.begin(), parent.begin(), pos);
        }

        // K Agents
        while (children.size() < K)
        {
            Node father, mother;
            if (EnableBestMateWorst)
            {
                auto parents = beta_getParent(parent, children.size() / 2);
                father = parents.first;
                mother = parents.second;
            }
            else
            {
                father = getNode(parent);
                mother = getNode(parent);
            }
            par child = father.makeChild(mother);
            children.push_back(child.first);
            children.push_back(child.second);
        }

        // Sort nodes by fitness (first - highest)
        sortNodes(children);

        if (EnableAdaptiveEvolutionRate)
        {
            if (BestFitness != children.begin()->value())
            {
                BestFitness = children.begin()->value();
                BestFitnessCounter = 0;
            }
            else
                ++BestFitnessCounter;
            beta_AdjustEvolutionRate(BestFitnessCounter);
        }

        if (TrackAllFitness)
        {
            vector<int> fitness;
            fitness.reserve(children.size());
            for (auto &x : children)
                fitness.push_back(x.value());
            all_fitness.push_back(fitness);
        }
    }

    // Output the result
    Node res = *children.begin();
    freopen(OutputPath, "w", stdout);
    if (IncludeTimeDate)
        cout << _time << '\t' << _date
             << endl
             << endl;
    cout << "> Pitch values" << endl;
    res.printPitch();
    cout << "> Pitch Names" << endl;
    res.printName();
    if (TrackAllFitness)
    {
        freopen(FitnessDataPath, "w", stdout);
        for (auto &f : all_fitness)
        {
            for (auto &x : f)
                cout << x << ',';
            cout << endl;
        }
    }
}
