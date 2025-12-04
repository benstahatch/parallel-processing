// random_forest_mpi.cpp
// Compile: mpicxx -O3 -std=c++17 random_forest_mpi.cpp -o rf_mpi
// Run: mpirun -np 4 ./rf_mpi
#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;
using Clock = chrono::high_resolution_clock;
using ll = long long;
struct Dataset {
    vector<vector<double>> X;
    vector<int> y;
    int n, d;
};

static std::mt19937 rng(123456);

Dataset make_synthetic(int n, int d, double noise=0.1) {
    Dataset ds;
    ds.n = n; ds.d = d;
    ds.X.assign(n, vector<double>(d));
    ds.y.assign(n, 0);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i=0;i<n;++i){
        double sum = 0;
        for (int j=0;j<d;++j){
            ds.X[i][j] = nd(rng);
            sum += ds.X[i][j] * ( (j%2==0) ? 1.0 : -0.7 );
        }
        double prob = 1.0/(1.0+exp(-sum));
        double r = std::uniform_real_distribution<double>(0,1)(rng);
        ds.y[i] = (r < prob*(1-noise) + (1-prob)*noise) ? 1 : 0;
    }
    return ds;
}

double gini_from_counts(int c0, int c1){
    int tot = c0 + c1;
    if(tot==0) return 0;
    double p0 = double(c0)/tot;
    double p1 = double(c1)/tot;
    return 1.0 - p0*p0 - p1*p1;
}

struct Node {
    bool is_leaf = true;
    int pred = 0;
    int feature = -1;
    double threshold = 0.0;
    Node *left = nullptr, *right = nullptr;
    ~Node(){ delete left; delete right; }
};

struct DecisionTree {
    int max_depth = 10;
    int min_samples_split = 2;
    int mtry; // number of features to consider each split
    Node *root = nullptr;
    DecisionTree(int mtry_): mtry(mtry_) {}
    ~DecisionTree(){ delete root; }
    void fit(const Dataset &ds, const vector<int>& indices){
        root = build_node(ds, indices, 0);
    }
    int predict_one(const vector<double>& x) const {
        Node *p = root;
        while(!p->is_leaf){
            if (x[p->feature] <= p->threshold) p = p->left;
            else p = p->right;
        }
        return p->pred;
    }
private:
    Node* build_node(const Dataset &ds, const vector<int>& indices, int depth){
        Node* node = new Node();
        int c0=0,c1=0;
        for(int idx: indices) (ds.y[idx]==0?c0:c1)++;
        node->pred = (c1 >= c0) ? 1 : 0;
        if (depth >= max_depth || (int)indices.size() < min_samples_split || c0==0 || c1==0){
            node->is_leaf = true;
            return node;
        }
        vector<int> features(ds.d);
        iota(features.begin(), features.end(), 0);
        shuffle(features.begin(), features.end(), rng);
        features.resize(min(mtry, ds.d));
        double best_gain = 0.0;
        int best_feat = -1;
        double best_thr = 0.0;
        vector<int> left_best, right_best;
        double parent_gini = gini_from_counts(c0, c1);
        for(int f: features){
            vector<pair<double,int>> vals;
            vals.reserve(indices.size());
            for(int idx: indices) vals.emplace_back(ds.X[idx][f], ds.y[idx]);
            sort(vals.begin(), vals.end(), [](auto &a, auto &b){ return a.first < b.first; });
            int left0=0,left1=0;
            int right0=0,right1=0;
            for(auto &p: vals) (p.second==0?right0:right1)++;
            for(size_t t=1;t<vals.size();++t){
                if (vals[t].first == vals[t-1].first){
                    if (vals[t].second==0) { --right0; ++left0; }
                    else { --right1; ++left1; }
                    continue;
                }
                if (vals[t-1].second==0){ --right0; ++left0; }
                else { --right1; ++left1; }
                int ls = left0 + left1;
                int rs = right0 + right1;
                if (ls < min_samples_split || rs < min_samples_split) continue;
                double g_left = gini_from_counts(left0,left1);
                double g_right = gini_from_counts(right0,right1);
                double gain = parent_gini - ( (double)ls/(ls+rs)*g_left + (double)rs/(ls+rs)*g_right );
                if (gain > best_gain){
                    best_gain = gain;
                    best_feat = f;
                    best_thr = 0.5*(vals[t].first + vals[t-1].first);
                    left_best.clear(); right_best.clear();
                    for(int idx: indices){
                        if (ds.X[idx][f] <= best_thr) left_best.push_back(idx); else right_best.push_back(idx);
                    }
                }
            }
        }
        if (best_feat == -1){
            node->is_leaf = true;
            return node;
        }
        node->is_leaf = false;
        node->feature = best_feat;
        node->threshold = best_thr;
        node->left = build_node(ds, left_best, depth+1);
        node->right = build_node(ds, right_best, depth+1);
        return node;
    }
};

struct RandomForestLocal {
    int n_trees_local;
    int max_depth;
    int min_samples_split;
    int mtry;
    vector<DecisionTree*> trees;
    RandomForestLocal(int n_trees_local_, int mtry_, int max_depth_=12, int min_samples_split_=2)
        : n_trees_local(n_trees_local_), max_depth(max_depth_), min_samples_split(min_samples_split_), mtry(mtry_)
    {
        trees.reserve(n_trees_local);
        for(int i=0;i<n_trees_local;++i) trees.push_back(nullptr);
    }
    ~RandomForestLocal(){ for(auto *t: trees) delete t; }
    void fit(const Dataset &ds){
        int n = ds.n;
        std::uniform_int_distribution<int> uid(0,n-1);
        for(int t=0;t<n_trees_local;++t){
            vector<int> sample_idx;
            sample_idx.reserve(n);
            for(int i=0;i<n;++i) sample_idx.push_back(uid(rng));
            DecisionTree* dt = new DecisionTree(mtry);
            dt->max_depth = max_depth;
            dt->min_samples_split = min_samples_split;
            dt->fit(ds, sample_idx);
            trees[t] = dt;
        }
    }
    // returns local votes (sum of predictions across local trees)
    vector<int> partial_predict_votes(const Dataset &ds){
        int n = ds.n;
        vector<int> votes(n,0);
        for(auto *t: trees){
            for(int i=0;i<n;++i) votes[i] += t->predict_one(ds.X[i]);
        }
        return votes;
    }
};

double accuracy(const vector<int>& a, const vector<int>& b){
    int n = a.size();
    int ok=0;
    for(int i=0;i<n;++i) if (a[i]==b[i]) ++ok;
    return double(ok)/n;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // params (same as serial for comparability)
    int n_train = 40000;
    int n_test  = 5000;
    int d = 20;
    int n_trees = 100;
    int mtry = max(1, (int)floor(sqrt(d)));
    if (rank==0) cout << "MPI Random Forest: n_train="<<n_train<<" n_test="<<n_test<<" d="<<d<<" n_trees="<<n_trees<<" mtry="<<mtry<<" ranks="<<size<<"\n";

    // create dataset deterministically on all ranks (simple approach)
    Dataset train = make_synthetic(n_train, d);
    Dataset test = make_synthetic(n_test, d);

    // split trees across ranks
    int base = n_trees / size;
    int rem = n_trees % size;
    int n_local = base + (rank < rem ? 1 : 0);

    RandomForestLocal rflocal(n_local, mtry, 15, 2);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    rflocal.fit(train);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    // local votes
    vector<int> local_votes = rflocal.partial_predict_votes(test);
    // reduce votes across ranks (sum)
    vector<int> global_votes(test.n,0);
    MPI_Reduce(local_votes.data(), global_votes.data(), test.n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    if (rank==0){
        vector<int> pred(test.n, 0);
        for(int i=0;i<test.n;++i) pred[i] = (global_votes[i] >= (n_trees+1)/2) ? 1 : 0;
        cout << "Train time (s): " << (t1 - t0) << "\n";
        cout << "Predict+Reduce time (s): " << (t2 - t1) << "\n";
        cout << "Test acc: " << accuracy(pred, test.y) << "\n";
    }
    MPI_Finalize();
    return 0;
}

