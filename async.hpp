ASYNC_BEGIN

template<typename _rtnT, typename ... _parasT> std::function<_rtnT(_parasT...)> capsulate_function(_rtnT(*func_val)(_parasT...))
{
    std::function<_rtnT(_parasT...)> func_temp = static_cast<_rtnT(*)(_parasT...)>(func_val);
    return func_temp;
}
template<typename _funcT, typename ... _parasT> auto package_function(_funcT &&func_val, _parasT &&...paras) -> std::shared_ptr<std::packaged_task<typename std::result_of<_funcT(_parasT...)>::type()>>
{
    using _rtnT = typename std::result_of<_funcT(_parasT...)>::type;
    return std::make_shared<std::packaged_task<_rtnT()>>(std::bind(std::forward<_funcT>(func_val), std::forward<_parasT>(paras)...));
}

template<typename _varT> class async_variable
{
public:
    async_variable(_varT init_val = _varT()) : value(init_val) {}
    async_variable(async_variable &src) : value(src._value()) {}
    void operator=(_varT val) { _value(val); }
    void operator=(async_variable &src) { _value(src._value()); }
    operator _varT() { return _value(); }
    _varT _value()
    {
        std::shared_lock<std::shared_mutex> lock(shrd_mtx);
        return value;
    }
    void _value(_varT tar_sgn)
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        value = tar_sgn;
    }
protected:
    mutable std::shared_mutex shrd_mtx;
    _varT value;
};

template<typename digitT, typename = std::enable_if_t<std::is_floating_point<digitT>::value || std::is_integral<digitT>::value>> class async_digit : public async_variable<digitT>
{
public:
    async_digit(digitT init_val = 0) : async_variable(init_val) {}
    async_digit(async_variable &src) : async_variable(src) {}
    void operator=(digitT val) { async_variable::operator=(val); }
    void operator=(async_variable &src) { async_variable::operator=(src); }
    void operator+=(async_variable &src)
    {
        static_assert(!std::is_same<digitT, bool>::value, "Type \"bool\" could not be incremented.");
        value += src.value;
    }
    void operator+=(digitT &src)
    {
        static_assert(!std::is_same<digitT, bool>::value, "Type \"bool\" could not be incremented.");
        value += src;
    }
    void operator-=(async_variable &src)
    {
        static_assert(!std::is_same<digitT, bool>::value, "Type \"bool\" could not be decremented.");
        value -= src.value;
    }
    void operator-=(digitT &src)
    {
        static_assert(!std::is_same<digitT, bool>::value, "Type \"bool\" could not be decremented.");
        value -= src;
    }
    async_digit &operator++() { increment(); return *this; }
    async_digit operator++(int) { auto temp = *this; increment(); return temp; }
    async_digit &operator--() { decrement(); return *this; }
    async_digit operator--(int) { auto temp = *this; decrement(); return temp; }
    void increment()
    {
        static_assert(!std::is_same<digitT, bool>::value, "Type \"bool\" could not be incremented.");
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        ++ value;
    }
    void decrement()
    {
        static_assert(!std::is_same<digitT, bool>::value, "Type \"bool\" could not be decremented.");
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        -- value;
    }
};

template<typename T> class async_queue
{
private:
    bagrt::net_link<T> ls_val;
    mutable std::shared_mutex tdmtx;
public:
    async_queue(async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = src.ls_val;
    }
    async_queue(async_queue &&src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = std::move(src.ls_val);
        src.reset();
    }
    void operator=(async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = src.ls_val;
    }
    void operator=(async_queue &&src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        ls_val = std::move(src.ls_val);
        src.reset();
    }
    bool operator==(async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        return (this->ls_val == src.ls_val);
    }
    bool operator!=(async_queue &src) { return !(*this != src); }
    friend std::ostream &operator<<(std::ostream &output, async_queue &src)
    {
        std::unique_lock<std::shared_mutex> lck(src.tdmtx);
        output << src.ls_val << std::endl;
        return output;
    }

    async_queue() {}
    uint64_t size()
    {
        std::shared_lock<std::shared_mutex> lck(tdmtx);
        return ls_val.size();
    }
    template<typename...args> bool en_queue(args &&...paras)
    {
        std::unique_lock<std::shared_mutex> lck(tdmtx);
        return ls_val.emplace_back(std::forward<args>(paras)...);
    }
    T de_queue()
    {
        std::unique_lock<std::shared_mutex> lck(tdmtx);
        return ls_val.erase(IDX_ZERO);
    }
    void reset()
    {
        std::unique_lock<std::shared_mutex> lck(tdmtx);
        ls_val.reset();
    }
    ~async_queue() { reset(); }
};

class async_batch
{
private:
    bagrt::net_queue<std::thread> td_set;
    bagrt::net_queue<std::function<void()>> tsk_set;
    async_digit<uint64_t> tsk_cnt = 0;
    async_digit<bool> stop = false;
    bagrt::net_queue<async_digit<bool>> proc_set;
    std::mutex td_mtx_tsk, td_mtx_proc;
    std::condition_variable cond_tsk, cond_proc;
    uint64_t asyn_batch_size = ASYNC_CORE_CNT;
public:
    async_batch(uint64_t batch_size = ASYNC_CORE_CNT) : asyn_batch_size(batch_size), td_set(batch_size), tsk_set(batch_size), proc_set(batch_size)
    {
        for(auto i=0; i<asyn_batch_size; ++i) td_set[i] = std::thread([this](int idx){ while(true)
        {
            {
                std::unique_lock<std::mutex> lkTsk(td_mtx_tsk);
                while(!(proc_set[idx] || stop)) cond_tsk.wait(lkTsk);
            }
            if(stop) return;
            tsk_set[idx]();
            -- tsk_cnt;
            {
                std::unique_lock<std::mutex> lkProc(td_mtx_proc);
                proc_set[idx] = false;
                cond_proc.notify_all();
            }
        }}, i);
    }
    template<typename _func, typename ... _para> auto set_task(uint64_t bat_idx, _func &&func_val, _para &&...args) -> std::future<typename std::result_of<_func(_para...)>::type>
    {
        using ret_type = typename std::result_of<_func(_para...)>::type;
        auto p_curr_task = package_function(std::forward<_func>(func_val), std::forward<_para>(args)...);
        std::future<ret_type> res = p_curr_task->get_future();
        if(bat_idx < asyn_batch_size)
        {
            std::unique_lock<std::mutex> lkProc(td_mtx_proc);
            while(proc_set[bat_idx] && !stop) cond_proc.wait(lkProc);
            tsk_set[bat_idx] = [p_curr_task]() { (*p_curr_task)(); };
        }
        else stop = true;
        ++ tsk_cnt;
        {
            std::unique_lock<std::mutex> lkTsk(td_mtx_tsk);
            proc_set[bat_idx] = true;
            cond_tsk.notify_all();
        }
        return res;
    }
    uint64_t task_cnt() { return tsk_cnt; }
    uint64_t batch_size() { return asyn_batch_size; }
    ~async_batch()
    {
        stop = true;
        cond_tsk.notify_all();
        for(auto i=0; i<td_set.size(); ++i) if(td_set[i].joinable()) td_set[i].join();
    }
};

class async_pool
{
private:
    bagrt::net_queue<std::thread> td_set;
    async_queue<std::function<void()>> tsk_set;
    std::mutex td_mtx;
    std::condition_variable cond;
    async_digit<bool> stop = false;
public:
    async_pool(uint64_t thread_size = ASYNC_CORE_CNT) : stop(false), td_set(thread_size) {
    for(auto i=0; i<td_set.size(); ++i) td_set[i] = std::thread([this]
    {
        while(true)
        {
            std::function<void()> curr_tsk;
            {
                std::unique_lock<std::mutex> lck(td_mtx);
                while(!(this->tsk_set.size() || stop)) cond.wait(lck);
                if(this->stop && !this->tsk_set.size()) return;
                curr_tsk = std::move(this->tsk_set.de_queue());
            }
            curr_tsk();
        }
    });}
    template<typename _funcT, typename ... _parasT> auto add_task(_funcT &&func_val, _parasT &&...paras) -> std::future<typename std::result_of<_funcT(_parasT ...)>::type>
    {
        // Thread function result type name (For deduce)
        using _rtnT = std::result_of<_funcT(_parasT...)>::type;
        // Function task package
        auto task = package_function(std::forward<_funcT>(func_val), std::forward<_parasT>(paras)...);
        // Get task result
        std::future<_rtnT> rtn = task->get_future();
        {
            // Mutex lock
            std::unique_lock<std::mutex> lock(td_mtx);
            if(stop) throw std::runtime_error("Stop thread pool.");
            // Add task
            tsk_set.en_queue([task](){ (*task)(); });
        }
        cond.notify_one();
        return rtn;
    }
    ~async_pool()
    {
        stop = true;
        cond.notify_all();
        for(auto i=0; i<td_set.size(); ++i) if(td_set[i].joinable()) td_set[i].join();
    }
};

class async_control
{
public:
    void thread_sleep()
    {
        std::unique_lock<std::mutex> lk(td_mtx);
        cond.wait(lk);
    }
    void thread_wake_all() { cond.notify_all(); }
    void thread_wake_one() { cond.notify_one(); }
private:
    std::mutex td_mtx;
    std::condition_variable cond;
};

class async_concurrent
{
public:
    async_concurrent(uint64_t batch_size = ASYNC_CORE_CNT) : async_batch_size(batch_size) {}
    void batch_thread_attach() { ctrl_batch.thread_sleep(); }
    void batch_thread_detach(std::function<void()> concurr_opt = []{ return; })
    {
        if((++proc_cnt) == async_batch_size)
        {
            concurr_opt();
            ctrl_main.thread_wake_one();
        }
    }
    void main_thread_deploy_batch_thread()
    {
        ctrl_batch.thread_wake_all();
        ctrl_main.thread_sleep();
        proc_cnt = 0;
    }
    void main_thread_exception() { ctrl_batch.thread_wake_all(); }
private:
    uint64_t async_batch_size = 0;
    async::async_digit<uint64_t> proc_cnt = 0;
    async_control ctrl_batch, ctrl_main;
};

template<typename funcT, typename ... argsT> void set_batch_thread(uint64_t iAsyncBatchSize, async::async_digit<uint64_t> &asyncCnt, funcT &&funcVal, argsT &&...paras)
{
    
    async::async_batch asyncBatch(iAsyncBatchSize);
    for(auto i=0; i<iAsyncBatchSize; ++i) asyncBatch.set_task(i, 
    [&asyncCnt, &funcVal]
    (int idx, argsT &&...args)
    {
        funcVal(idx, std::forward<argsT>(args)...);
        ++ asyncCnt;
    }, i, std::forward<argsT>(paras)...);
    while(asyncCnt != iAsyncBatchSize);
}

ASYNC_END