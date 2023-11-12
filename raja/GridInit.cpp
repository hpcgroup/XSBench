#include "XSbench_header.h"

// Moves all required data structures to the GPU's memory space
SimulationData move_simulation_data_to_device( Inputs in, int mype, SimulationData SD )
{
	if(mype == 0) printf("Allocating and moving simulation data to GPU memory space...\n");

	////////////////////////////////////////////////////////////////////////////////
	// SUMMARY: Simulation Data Structure Manifest for "SD" Object
	// Here we list all heap arrays (and lengths) in SD that would need to be
	// offloaded manually if using an accelerator with a seperate memory space
	////////////////////////////////////////////////////////////////////////////////
	// int * num_nucs;                     // Length = length_num_nucs;
	// double * concs;                     // Length = length_concs
	// int * mats;                         // Length = length_mats
	// double * unionized_energy_array;    // Length = length_unionized_energy_array
	// int * index_grid;                   // Length = length_index_grid
	// NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
	// 
	// Note: "unionized_energy_array" and "index_grid" can be of zero length
	//        depending on lookup method.
	//
	// Note: "Lengths" are given as the number of objects in the array, not the
	//       number of bytes.
	////////////////////////////////////////////////////////////////////////////////
	size_t sz;
	size_t total_sz = 0;

	// Shallow copy of CPU simulation data to GPU simulation data
	SimulationData GSD = SD;

    auto& rm = umpire::ResourceManager::getInstance();
	umpire::Allocator allocator = rm.getAllocator("DEVICE");

	// Move data to GPU memory space
	sz = GSD.length_num_nucs * sizeof(int);
    GSD.num_nucs = static_cast<int*>(allocator.allocate(sz));
    rm.copy(GSD.num_nucs, SD.num_nucs);
	total_sz += sz;

	sz = GSD.length_concs * sizeof(double);
    GSD.concs = static_cast<double*>(allocator.allocate(sz));
    rm.copy(GSD.concs, SD.concs);
	total_sz += sz;

	sz = GSD.length_mats * sizeof(int);
    GSD.mats = static_cast<int*>(allocator.allocate(sz));
    rm.copy(GSD.mats, SD.mats);
	total_sz += sz;
	
	sz = GSD.length_unionized_energy_array * sizeof(double);
    GSD.unionized_energy_array = static_cast<double*>(allocator.allocate(sz));
    rm.copy(GSD.unionized_energy_array, SD.unionized_energy_array);
	total_sz += sz;

	sz = GSD.length_index_grid * sizeof(int);
    GSD.index_grid = static_cast<int*>(allocator.allocate(sz));
    rm.copy(GSD.index_grid, SD.index_grid);
	total_sz += sz;

	sz = GSD.length_nuclide_grid * sizeof(NuclideGridPoint);
    GSD.nuclide_grid = static_cast<NuclideGridPoint*>(allocator.allocate(sz));
    rm.copy(GSD.nuclide_grid, SD.nuclide_grid);
	total_sz += sz;
	
	// Allocate verification array on device. This structure is not needed on CPU, so we don't
	// have to copy anything over.
	sz = in.lookups * sizeof(unsigned long);
    GSD.verification = static_cast<unsigned long *>(allocator.allocate(sz));
	total_sz += sz;
	GSD.length_verification = in.lookups;
	
	if(mype == 0 ) printf("GPU Intialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0 );

	return GSD;

}

// Release device memory
void release_device_memory(SimulationData GSD) {
    auto& rm = umpire::ResourceManager::getInstance();
	umpire::Allocator allocator = rm.getAllocator("DEVICE");	

	allocator.deallocate(GSD.num_nucs);
	allocator.deallocate(GSD.concs);
	allocator.deallocate(GSD.mats);
	allocator.deallocate(GSD.unionized_energy_array);
	allocator.deallocate(GSD.nuclide_grid);
	allocator.deallocate(GSD.verification);
}

SimulationData grid_init_do_not_profile( Inputs in, int mype )
{
	// Structure to hold all allocated simuluation data arrays
	SimulationData SD;

	// Keep track of how much data we're allocating
	size_t nbytes = 0;
	
	// Set the initial seed value
	uint64_t seed = 42;	

	////////////////////////////////////////////////////////////////////
	// Initialize Nuclide Grids
	////////////////////////////////////////////////////////////////////
	
	if(mype == 0) printf("Intializing nuclide grids...\n");

	// First, we need to initialize our nuclide grid. This comes in the form
	// of a flattened 2D array that hold all the information we need to define
	// the cross sections for all isotopes in the simulation. 
	// The grid is composed of "NuclideGridPoint" structures, which hold the
	// energy level of the grid point and all associated XS data at that level.
	// An array of structures (AOS) is used instead of
	// a structure of arrays, as the grid points themselves are accessed in 
	// a random order, but all cross section interaction channels and the
	// energy level are read whenever the gridpoint is accessed, meaning the
	// AOS is more cache efficient.
    
    auto& rm = umpire::ResourceManager::getInstance();
	umpire::Allocator allocator = rm.getAllocator("HOST");
	
	// Initialize Nuclide Grid
	SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
	SD.nuclide_grid     = static_cast<NuclideGridPoint *>(allocator.allocate( SD.length_nuclide_grid * sizeof(NuclideGridPoint)));
	assert(SD.nuclide_grid != NULL);
	nbytes += SD.length_nuclide_grid * sizeof(NuclideGridPoint);
	for( int i = 0; i < SD.length_nuclide_grid; i++ )
	{
		SD.nuclide_grid[i].energy        = LCG_random_double(&seed);
		SD.nuclide_grid[i].total_xs      = LCG_random_double(&seed);
		SD.nuclide_grid[i].elastic_xs    = LCG_random_double(&seed);
		SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
		SD.nuclide_grid[i].fission_xs    = LCG_random_double(&seed);
		SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
	}

	// Sort so that each nuclide has data stored in ascending energy order.
	for( int i = 0; i < in.n_isotopes; i++ )
		qsort( &SD.nuclide_grid[i*in.n_gridpoints], in.n_gridpoints, sizeof(NuclideGridPoint), NGP_compare);
	
	// error debug check
	/*
	for( int i = 0; i < in.n_isotopes; i++ )
	{
		printf("NUCLIDE %d ==============================\n", i);
		for( int j = 0; j < in.n_gridpoints; j++ )
			printf("E%d = %lf\n", j, SD.nuclide_grid[i * in.n_gridpoints + j].energy);
	}
	*/
	

	////////////////////////////////////////////////////////////////////
	// Initialize Acceleration Structure
	////////////////////////////////////////////////////////////////////
	
	if( in.grid_type == NUCLIDE )
	{
		SD.length_unionized_energy_array = 0;
		SD.length_index_grid = 0;
	}
	
	if( in.grid_type == UNIONIZED )
	{
		if(mype == 0) printf("Intializing unionized grid...\n");

		// Allocate space to hold the union of all nuclide energy data
		SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
		SD.unionized_energy_array = static_cast<double *>(allocator.allocate( SD.length_unionized_energy_array * sizeof(double)));
		assert(SD.unionized_energy_array != NULL );
		nbytes += SD.length_unionized_energy_array * sizeof(double);

		// Copy energy data over from the nuclide energy grid
		for( int i = 0; i < SD.length_unionized_energy_array; i++ )
			SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;

		// Sort unionized energy array
		qsort( SD.unionized_energy_array, SD.length_unionized_energy_array, sizeof(double), double_compare);

		// Allocate space to hold the acceleration grid indices
		SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
		SD.index_grid = static_cast<int *>(allocator.allocate( SD.length_index_grid * sizeof(int)));
		assert(SD.index_grid != NULL);
		nbytes += SD.length_index_grid * sizeof(int);

		// Generates the double indexing grid
		int * idx_low = (int *) calloc( in.n_isotopes, sizeof(int));
		assert(idx_low != NULL );
		double * energy_high = (double *) malloc( in.n_isotopes * sizeof(double));
		assert(energy_high != NULL );

		for( int i = 0; i < in.n_isotopes; i++ )
			energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + 1].energy;

		for( long e = 0; e < SD.length_unionized_energy_array; e++ )
		{
			double unionized_energy = SD.unionized_energy_array[e];
			for( long i = 0; i < in.n_isotopes; i++ )
			{
				if( unionized_energy < energy_high[i]  )
					SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
				else if( idx_low[i] == in.n_gridpoints - 2 )
					SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
				else
				{
					idx_low[i]++;
					SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
					energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + idx_low[i] + 1].energy;	
				}
			}
		}

		free(idx_low);
		free(energy_high);
	}

	if( in.grid_type == HASH )
	{
		if(mype == 0) printf("Intializing hash grid...\n");
		SD.length_unionized_energy_array = 0;
		SD.length_index_grid  = in.hash_bins * in.n_isotopes;

		SD.index_grid = static_cast<int *>(allocator.allocate( SD.length_index_grid * sizeof(int)));
		assert(SD.index_grid != NULL);
		nbytes += SD.length_index_grid * sizeof(int);

		double du = 1.0 / in.hash_bins;

		// For each energy level in the hash table
		for( long e = 0; e < in.hash_bins; e++ )
		{
			double energy = e * du;

			// We need to determine the bounding energy levels for all isotopes
			for( long i = 0; i < in.n_isotopes; i++ )
			{
				SD.index_grid[e * in.n_isotopes + i] = grid_search_nuclide( in.n_gridpoints, energy, SD.nuclide_grid + i * in.n_gridpoints, 0, in.n_gridpoints-1);
			}
		}
	}

	////////////////////////////////////////////////////////////////////
	// Initialize Materials and Concentrations
	////////////////////////////////////////////////////////////////////
	if(mype == 0) printf("Intializing material data...\n");
	
	// Set the number of nuclides in each material
	SD.num_nucs  = load_num_nucs(in.n_isotopes);
	SD.length_num_nucs = 12; // There are always 12 materials in XSBench

	// Intialize the flattened 2D grid of material data. The grid holds
	// a list of nuclide indices for each of the 12 material types. The
	// grid is allocated as a full square grid, even though not all
	// materials have the same number of nuclides.
	SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
	SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;

	// Intialize the flattened 2D grid of nuclide concentration data. The grid holds
	// a list of nuclide concentrations for each of the 12 material types. The
	// grid is allocated as a full square grid, even though not all
	// materials have the same number of nuclides.
	SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
	SD.length_concs = SD.length_mats;


    SD.verification = static_cast<unsigned long *>(allocator.allocate(in.lookups * sizeof(unsigned long)));
	nbytes += in.lookups * sizeof(unsigned long);
	SD.length_verification = in.lookups;

	if(mype == 0) printf("Intialization complete. Allocated %.0lf MB of data on CPU.\n", nbytes/1024.0/1024.0 );

	return SD;
}
