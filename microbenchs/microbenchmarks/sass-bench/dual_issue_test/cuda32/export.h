
typedef struct inst_latency_info {
	int t_reissue;
	int t_iadd;
	int t_dfma;
	int t_branch;
	int t_ld_sm128_raw;
	int t_st_sm128_raw;
	int t_ld_sm128_war;
	int t_st_sm128_war;
	int t_ld_gm128_L2bypass_raw;
	int t_ld_gm128_L2hit_raw;
	int t_ld_gm128_war;
};

typedef struct dual_issue_info {
	int t_iadd[5];
	int t_fadd[5];
	int t_fmul[5];
	int t_dfma[5];
};
	
void measure_inst_latency(inst_latency_info *info);
void disp_inst_latency(void);
 
void measure_dual_issue(dual_issue_info *info);
void disp_dual_issue(void); 

