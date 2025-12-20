#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <set>
#include <sys/resource.h>

using namespace std;

// -----------------------------------------------------------------------------
// MEMORY USAGE (MB)
// -----------------------------------------------------------------------------
double getMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

// -----------------------------------------------------------------------------
// TREE NODE
// -----------------------------------------------------------------------------
struct TreeNode {
    bool isLeaf;
    int predictedClass;
    int featureIndex;
    double threshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode() :
        isLeaf(false), predictedClass(0),
        featureIndex(-1), threshold(0.0),
        left(nullptr), right(nullptr) {}

    ~TreeNode() { delete left; delete right; }
};

// -----------------------------------------------------------------------------
// DECISION TREE
// -----------------------------------------------------------------------------
class DecisionTree {
private:
    int maxDepth;
    int minSamplesSplit;
    TreeNode* root;
    mt19937 rng;

    double giniImpurity(const vector<int>& labels) {
        if (labels.empty()) return 0.0;

        map<int,int> counts;
        for (int l : labels) counts[l]++;

        double g = 1.0;
        for (auto &p : counts) {
            double prob = (double)p.second / labels.size();
            g -= prob * prob;
        }
        return g;
    }

    int mostCommonClass(const vector<int>& labels) {
        map<int,int> counts;
        for (int l : labels) counts[l]++;
        int best = 0, bestCount = 0;

        for (auto &p : counts) {
            if (p.second > bestCount) {
                bestCount = p.second;
                best = p.first;
            }
        }
        return best;
    }

    void splitData(const vector<vector<double>>& X, const vector<int>& y,
                   int f, double thr,
                   vector<vector<double>>& Xl, vector<int>& yl,
                   vector<vector<double>>& Xr, vector<int>& yr)
    {
        for (int i = 0; i < X.size(); i++) {
            if (X[i][f] <= thr) { Xl.push_back(X[i]); yl.push_back(y[i]); }
            else               { Xr.push_back(X[i]); yr.push_back(y[i]); }
        }
    }

    bool findBestSplit(const vector<vector<double>>& X, const vector<int>& y,
                       const vector<int>& features, int& bestF, double& bestT)
    {
        double bestGain = -1.0;
        double parent = giniImpurity(y);
        int m = y.size();

        for (int f : features) {
            set<double> uniq;
            for (auto &r : X) uniq.insert(r[f]);

            for (double thr : uniq) {
                vector<vector<double>> Xl, Xr;
                vector<int> yl, yr;
                splitData(X, y, f, thr, Xl, yl, Xr, yr);

                if (yl.empty() || yr.empty()) continue;

                double wl = (double)yl.size() / m;
                double wr = 1.0 - wl;

                double gain = parent -
                    (wl * giniImpurity(yl) + wr * giniImpurity(yr));

                if (gain > bestGain) {
                    bestGain = gain;
                    bestF = f;
                    bestT = thr;
                }
            }
        }
        return bestGain > 0;
    }

    TreeNode* buildTree(const vector<vector<double>>& X,
                        const vector<int>& y,
                        int depth,
                        const vector<int>& features)
    {
        TreeNode* node = new TreeNode();

        if (depth >= maxDepth || y.size() < minSamplesSplit ||
            giniImpurity(y) == 0.0)
        {
            node->isLeaf = true;
            node->predictedClass = mostCommonClass(y);
            return node;
        }

        int bestF; double bestT;
        if (!findBestSplit(X, y, features, bestF, bestT)) {
            node->isLeaf = true;
            node->predictedClass = mostCommonClass(y);
            return node;
        }

        vector<vector<double>> Xl, Xr;
        vector<int> yl, yr;
        splitData(X, y, bestF, bestT, Xl, yl, Xr, yr);

        node->isLeaf = false;
        node->featureIndex = bestF;
        node->threshold = bestT;
        node->left = buildTree(Xl, yl, depth+1, features);
        node->right = buildTree(Xr, yr, depth+1, features);

        return node;
    }

    int predictSample(const vector<double>& x, TreeNode* n) {
        if (n->isLeaf) return n->predictedClass;
        return (x[n->featureIndex] <= n->threshold)
            ? predictSample(x, n->left)
            : predictSample(x, n->right);
    }

public:
    DecisionTree(int maxDepth=10,int minSplit=2,int seed=42)
        : maxDepth(maxDepth), minSamplesSplit(minSplit), root(nullptr), rng(seed) {}

    ~DecisionTree() { delete root; }

    void fit(const vector<vector<double>>& X, const vector<int>& y,
             const vector<int>& features)
    {
        root = buildTree(X, y, 0, features);
    }

    vector<int> predict(const vector<vector<double>>& X) {
        vector<int> p;
        for (auto& r : X) p.push_back(predictSample(r, root));
        return p;
    }
};

// -----------------------------------------------------------------------------
// RANDOM FOREST
// -----------------------------------------------------------------------------
class RandomForest {
public:
    int nTrees, maxDepth, minSplit, maxFeatures;
    bool bootstrap;
    vector<DecisionTree*> trees;
    mt19937 rng;

    RandomForest(int n=100,int md=10,int ms=2,int mf=-1,bool bs=true,int seed=42)
        : nTrees(n), maxDepth(md), minSplit(ms), maxFeatures(mf),
          bootstrap(bs), rng(seed) {}

    ~RandomForest() { for (auto t : trees) delete t; }

    vector<int> chooseFeatures(int nF) {
        vector<int> f(nF);
        iota(f.begin(), f.end(), 0);
        shuffle(f.begin(), f.end(), rng);
        int k = (maxFeatures==-1)? sqrt(nF) : maxFeatures;
        return vector<int>(f.begin(), f.begin()+k);
    }

    void bootstrapSample(const vector<vector<double>>& X, const vector<int>& y,
                         vector<vector<double>>& Xs, vector<int>& ys)
    {
        uniform_int_distribution<int> d(0, X.size()-1);
        for (int i=0;i<X.size();i++){
            int idx = d(rng);
            Xs.push_back(X[idx]);
            ys.push_back(y[idx]);
        }
    }

    void fit(const vector<vector<double>>& X,const vector<int>& y){
        int nF = X[0].size();
        if (maxFeatures==-1) maxFeatures = sqrt(nF);

        for (int i=0;i<nTrees;i++){
            vector<vector<double>> Xs;
            vector<int> ys;

            bootstrapSample(X,y,Xs,ys);
            vector<int> feat = chooseFeatures(nF);

            DecisionTree* t = new DecisionTree(maxDepth,minSplit,42+i);
            t->fit(Xs,ys,feat);
            trees.push_back(t);

            if ((i+1)%10==0)
                cout<<"Trained "<<(i+1)<<"/"<<nTrees<<" trees\n";
        }
    }

    vector<int> predict(const vector<vector<double>>& X){
        vector<vector<int>> all;
        for (auto t : trees) all.push_back(t->predict(X));

        vector<int> finalP(X.size());
        for (int i=0;i<X.size();i++){
            map<int,int> vote;
            for (auto &v:all) vote[v[i]]++;
            int best=0,bc=0;
            for (auto &p:vote)
                if (p.second>bc){bc=p.second;best=p.first;}
            finalP[i]=best;
        }
        return finalP;
    }
};

// -----------------------------------------------------------------------------
// CSV + SCALING
// -----------------------------------------------------------------------------
void readCSV(const string& fn, vector<vector<double>>& X, vector<int>& y){
    ifstream f(fn);
    string line; bool skip=true;
    while (getline(f,line)){
        if (skip){ skip=false; continue; }
        stringstream ss(line); string v; vector<double> r;
        while (getline(ss,v,',')) r.push_back(stod(v));
        y.push_back((int)r.back()); r.pop_back(); X.push_back(r);
    }
}

void minMaxScale(vector<vector<double>>& X){
    int F = X[0].size();
    for (int j=0;j<F;j++){
        double mn=X[0][j], mx=X[0][j];
        for (int i=1;i<X.size();i++){
            mn=min(mn,X[i][j]); mx=max(mx,X[i][j]);
        }
        for (int i=0;i<X.size();i++){
            X[i][j] = (mx!=mn) ? (X[i][j]-mn)/(mx-mn) : 0.0;
        }
    }
}

// -----------------------------------------------------------------------------
// FULL METRICS
// -----------------------------------------------------------------------------
void printMetrics(const vector<int>& y, const vector<int>& p){

    int TP=0,TN=0,FP=0,FN=0;
    for (int i=0;i<y.size();i++){
        if (p[i]==1 && y[i]==1) TP++;
        else if (p[i]==0 && y[i]==0) TN++;
        else if (p[i]==1 && y[i]==0) FP++;
        else FN++;
    }

    double acc = (double)(TP+TN)/(TP+TN+FP+FN);
    double prec = (TP+FP)? (double)TP/(TP+FP) : 0;
    double rec  = (TP+FN)? (double)TP/(TP+FN) : 0;
    double f1   = (prec+rec)? 2*prec*rec/(prec+rec) : 0;

    cout << "\n---- Metrics ----\n";
    cout << "TP: " << TP << "  TN: " << TN << "\n";
    cout << "FP: " << FP << "  FN: " << FN << "\n";

    cout << "\n===== FULL MODEL METRICS =====\n";
    cout << "Accuracy:    " << acc * 100 << "%\n";
    cout << "Precision:   " << prec * 100 << "%\n";
    cout << "Recall:      " << rec * 100 << "%\n";
    cout << "F1 Score:    " << f1 * 100 << "%\n";
    cout << "================================\n";
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(){

    cout<<"============================================================\n";
    cout<<"Serial Random Forest\n";
    cout<<"============================================================\n";

    vector<vector<double>> X;
    vector<int> y;

    readCSV("diabetes.csv",X,y);
    cout<<"Dataset loaded: "<<X.size()<<" samples, "<<X[0].size()<<" features\n";

    minMaxScale(X);

    auto t0 = chrono::high_resolution_clock::now();

    RandomForest rf(100,10,2,-1,true,42);
    rf.fit(X,y);

    auto t1 = chrono::high_resolution_clock::now();
    double Tserial = chrono::duration<double>(t1-t0).count();

    vector<int> pred = rf.predict(X);
    double mem = getMemoryUsageMB();

    cout<<"\n---------------------------------------------------\n";
    cout<<"T_serial (seconds): "<<Tserial<<"\n";
    cout<<"Memory Usage (MB): "<<mem<<"\n";
    cout<<"---------------------------------------------------\n";

    printMetrics(y,pred);

    return 0;
}
