#pragma once
namespace omps {
	/// <summary>
	/// omp interface
	/// </summary>
	class Iomp
	{
	public:
		static int getMaxCanUseThreadCnt() noexcept;
		static int getMaxProcessorCnt() noexcept;
		///
	};
}


