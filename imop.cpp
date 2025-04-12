#include "Iomp.h"

#include <omp.h>
namespace omps {
	int Iomp::getMaxCanUseThreadCnt() noexcept {
		return omp_get_max_threads();
	}
	int Iomp::getMaxProcessorCnt() noexcept
	{
		return omp_get_num_procs();
	}
}