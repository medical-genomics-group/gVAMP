#pragma once
#include <string>
#include <vector>

class Options {

public:
    Options() = default; // the compiler automatically generates the default constructor if it does not declare its own
    Options(int argc, char** argv) {
        read_command_line_options(argc, argv);
        check_options();
        //if (!predict())
        //    read_group_mixture_file();
    }

    void read_command_line_options(int argc, char** argv);

    std::string get_bed_file() const { return bed_file; }
    std::string get_bed_file_test() const { return bed_file_test; }
    //std::string get_dim_file() const { return dim_file; }
    std::string get_bim_file() const { return bim_file; }
    std::string get_estimate_file() const { return estimate_file; }
    std::string get_cov_file() const { return cov_file; }
    //std::string get_ref_bim_file() const { return ref_bim_file; }
    //std::string get_group_index_file() const { return group_index_file; }
    //std::string get_group_mixture_file() const { return group_mixture_file; }

    std::string get_out_dir() const { return out_dir; }
    std::string get_out_name() const { return out_name; }
    std::string get_model() const { return model; }
    std::string get_run_mode() const { return run_mode; }

    double get_stop_criteria_thr() const { return stop_criteria_thr; }
    double get_EM_err_thr() const { return EM_err_thr; }
    double get_rho() const { return rho; }
    double get_probit_var() const { return probit_var; }
    

    unsigned int get_EM_max_iter() const { return EM_max_iter; }
    unsigned int get_CG_max_iter() const { return CG_max_iter; }
    unsigned int get_Mt()  const { return Mt; }
    unsigned int get_Mt_test()  const { return Mt_test; }
    unsigned int get_N() const { return N; }
    unsigned int get_N_test() const { return N_test; }
    unsigned int get_num_mix_comp() const { return num_mix_comp; }
    unsigned int get_use_lmmse_damp() const { return use_lmmse_damp; }
    unsigned int get_use_XXT_denoiser() const { return use_XXT_denoiser; } 
    unsigned int get_store_pvals() const { return store_pvals; }
    unsigned int get_CV() const { return CV; }
    unsigned int get_C() const { return C; }
    unsigned int get_redglob() const { return redglob; }
    unsigned int get_learn_vars() const { return learn_vars; }
    double get_h2() const { return h2; }

    std::vector<double> get_vars() const { return vars; } 
    std::vector<double> get_probs() const { return probs; }

    std::vector<int> get_test_iter_range() const { return test_iter_range; }

    const std::vector<std::string>& get_phen_files() const { return phen_files; }
    const std::vector<std::string>& get_phen_files_test() const { return phen_files_test; }

    const std::vector<std::string>& get_true_signal_files() const { return true_signal_files; }

    void list_phen_files() const;
    int  count_phen_files() const { return phen_files.size(); }
    int  count_phen_files_test() const { return phen_files_test.size(); }
    //int  get_verbosity() const { return verbosity; }
    //bool verbosity_level(const int level) const { return level > get_verbosity() ? false : true; }

    unsigned int get_iterations() const { return iterations; }
    //int get_ngroups()   const { return ngroups; }
    //int get_nmixtures() const { return nmixtures; }
    //const std::vector<std::vector<double>>& get_cva()  const { return cva;  }
    //const std::vector<std::vector<double>>& get_cvai() const { return cvai; }

    void set_probit_var(double v) { probit_var = v; }


private:
    std::string bed_file = "";
    std::string bed_file_test = "";
    std::string estimate_file = "";
    std::string cov_file = "";
    //std::string dim_file = "";
    std::string run_mode = "";
    std::string bim_file = "";
    //std::string ref_bim_file = "";
    //std::string group_index_file   = "";
    //std::string group_mixture_file = "";
    std::string out_dir = "";
    std::string out_name = "";
    std::string model = "linear";

    double stop_criteria_thr = 1e-4;
    double EM_err_thr = 1e-2;
    unsigned int EM_max_iter = 2;
    unsigned int CG_max_iter = 60;
    unsigned int Mt;
    unsigned int N;
    unsigned int N_test;
    unsigned int Mt_test;
    unsigned int num_mix_comp;
    unsigned int store_pvals = 0;
    unsigned int use_lmmse_damp = 0;
    unsigned int use_XXT_denoiser = 0;
    unsigned int learn_vars = 1;
    double alpha = 1; 
    unsigned int CV;
    unsigned int redglob = 0;
    unsigned int C = 0;
    double probit_var = 1;


    // unsigned int meth_imp = 0;

    std::vector<double> vars;
    std::vector<double> probs;
    std::vector<int> test_iter_range = std::vector<int>(2, -1);

    double rho = 0.98;
    double h2 = 0.5;
    //int verbosity = 0;
    unsigned int iterations = 1;

    void fail_if_last(char** argv, const int i);
    std::vector<std::string> phen_files;
    std::vector<std::string> phen_files_test;
    std::vector<std::string> true_signal_files;
    //int ngroups = 0;
    //void _set_ngroups(const int i) { ngroups = i; }
    //int nmixtures = 0;
    //void _set_nmixtures(const int i) { nmixtures = i; }

    void check_options();
    //void read_group_mixture_file();
    //std::vector<std::vector<double>> cva, cvai;
};