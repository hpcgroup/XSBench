#ifndef XSBENCH_SHARED_HEADER_H
#define XSBENCH_SHARED_HEADER_H

// Header for shared utilities across XSBench versions

typedef struct{
        int nthreads;
        long n_isotopes;
        long n_gridpoints;
        int lookups;
        char * HM;
        int grid_type; // 0: Unionized Grid (default)    1: Nuclide Grid
        int hash_bins;
        int particles;
        int simulation_method;
        int binary_mode;
        int kernel_id;
	int num_iterations;
} Inputs;

typedef struct{
	double device_to_host_time;
	double kernel_time;
	double host_to_device_time;
} Profile;

inline void print_profile(Profile profile, Inputs in) {
	printf("host_to_device_ms,kernel_ms,device_to_host_ms,num_iterations\n");
	printf("%f,%f,%f,%d\n",
	       profile.host_to_device_time*1000,
	       profile.kernel_time*1000,
	       profile.device_to_host_time*1000,
	       in.num_iterations);
}

#endif // XSBENCH_SHARED_HEADER_H
