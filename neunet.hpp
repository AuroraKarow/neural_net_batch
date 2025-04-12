NEUNET_BEGIN

struct NetOutputDataBatch
{
    set<uint64_t> setCurrOutputLbl;
    set<vect> setBatchOutput;
    NetOutputDataBatch() {}
    NetOutputDataBatch(NetOutputDataBatch &netSrc) { *this = netSrc; }
    NetOutputDataBatch(NetOutputDataBatch &&netSrc) { *this = std::move(netSrc); }
    void operator=(NetOutputDataBatch &netSrc)
    {
        setCurrOutputLbl = netSrc.setCurrOutputLbl;
        setBatchOutput = netSrc.setBatchOutput;
    }
    void operator=(NetOutputDataBatch &&netSrc)
    {
        setCurrOutputLbl = std::move(netSrc.setCurrOutputLbl);
        setBatchOutput = std::move(netSrc.setBatchOutput);
    }
    void Reset() { setCurrOutputLbl.reset(); setBatchOutput.reset(); }
    ~NetOutputDataBatch() { Reset(); }
};

class NetBase
{
protected:
    virtual void ValueAssign(NetBase &netSrc)
    { 
        dNetAcc = netSrc.dNetAcc; iNetBatchSize = netSrc.iNetBatchSize; iCurrLayerNo = netSrc.iCurrLayerNo; iConcurrLayerTask = netSrc.iConcurrLayerTask;
    }
    virtual bool ForwProp(set<vect> &setInput) { return true; }
    virtual bool BackProp(set<vect> &setInput) { return true; }
    virtual bool ForwProp(vect &vecInput) { return true; }
    virtual bool BackProp(vect &vecInput) { return true; }
    virtual bool Deduce(vect &vecInput) { return true; }
    void NeuronInit() {}
    void UpdatePara() {}
    bool RunLinear() { return true; }
    bool RunThread() { return true; }
public:
    virtual void ValueCopy(NetBase &netSrc) { ValueAssign(netSrc); seqLayer = netSrc.seqLayer; }
    virtual void ValueMove(NetBase &&netSrc) { ValueAssign(netSrc); seqLayer = std::move(netSrc.seqLayer); }
    NetBase(NetBase &netSrc) { ValueCopy(netSrc); }
    NetBase(NetBase &&netSrc) { ValueMove(std::move(netSrc)); }
    void operator=(NetBase &netSrc) { ValueCopy(netSrc); }
    void operator=(NetBase &&netSrc) { ValueMove(std::move(netSrc)); }
    
    NetBase(double dNetAcc = 1e-5, uint64_t iBatchSize = 0, bool bUseMultiThread = true) : dNetAcc(dNetAcc), iNetBatchSize(iBatchSize), bMultiThreadMode(bUseMultiThread), asyncConcurr(iBatchSize), asyncBatch(bUseMultiThread?iBatchSize:IDX_ZERO), asyncPool(bUseMultiThread?IDX_SGL:IDX_ZERO) {}
    /* Add layer with corresponding parameters
     * - LAYER_ACT
     * uint64_t iActFuncType
     * - LAYER_FC
     * uint64_t iOutputLnCnt = 0
     * double dLearnRate = 0
     * double dRandBoundryFirst = 0
     * double dRandBoundrySecond = 0
     * double dRandBoundryAcc = 1e-5
     * - LAYER_CONV_IM2COL
     * uint64_t iKernelAmt = 0
     * uint64_t iKernelLnCnt = 0
     * uint64_t iKernelColCnt = 0
     * uint64_t iLnStride = 0
     * uint64_t iColStride = 0
     * double dLearnRate = 0
     * uint64_t iLnDilation = 0
     * uint64_t iColDilation = 0
     * double dRandBoundryFirst = 0
     * double dRandBoundrySecond = 0
     * double dRandBoundryAcc = 1e-5
     * - LAYER_CONV_BN_IM2COL
     * double dShift = 0
     * double dScale = 1
     * double dLearnRate = 0
     * double dDmt = 1e-8
     * - LAYER_POOL_IM2COL
     * uint64_t iPoolTypeVal = POOL_MAX_IM2COL
     * uint64_t iFilterLnCnt = 0
     * uint64_t iFilterColCnt = 0
     * uint64_t iLnStride = 0
     * uint64_t iColStride = 0
     * uint64_t iLnDilation = 0
     * uint64_t iColDilation = 0
     * - LAYER_PAD_CROP_IM2COL
     * bool bPadMode = true
     * uint64_t iAlterTop = 0
     * uint64_t iAlterRight = 0
     * uint64_t iAlterBottom = 0
     * uint64_t iAlterLeft = 0
     * uint64_t iAlterLnDistance = 0
     * uint64_t iAlterColDistance = 0
     * - LAYER_TRANS_IM2COL
     * uint64_t iChannLnCnt = 0
     * uint64_t iChannColCnt = 0
     * uint64_t iChannCnt = 1
     */
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) { return seqLayer.emplace_back(std::make_shared<LayerType>(std::forward<Args>(pacArgs)...)); }
    virtual uint64_t Depth() { return seqLayer.size(); }
    bool Run()
    {
        NeuronInit();
        if(bMultiThreadMode) return RunThread();
        else return RunLinear();
    }
    virtual void Reset() { seqLayer.reset(); }
    ~NetBase() { Reset(); }
protected:
    double dNetAcc = 1e-5;
    uint64_t iNetBatchSize = 0, iCurrLayerNo = 0, iConcurrLayerTask = LAYER_OPT_UPDATE_PARA;
    bool bMultiThreadMode = true;
    async::async_concurrent asyncConcurr;
    async::async_control asyncMainCtrl;
    async::async_batch asyncBatch;
    async::async_pool asyncPool;
    NET_SEQ<LAYER_PTR> seqLayer;
};

class NetMNISTIm2Col final : public NetBase
{
public:
    void ValueCopy(NetMNISTIm2Col &netSrc) { mapBNData = netSrc.mapBNData; }
    void ValueMove(NetMNISTIm2Col &&netSrc) { mapBNData = std::move(netSrc.mapBNData); }
    NetMNISTIm2Col(NetMNISTIm2Col &netSrc) : NetBase(netSrc) { ValueCopy(netSrc); }
    NetMNISTIm2Col(NetMNISTIm2Col &&netSrc) : NetBase(std::move(netSrc)) { ValueMove(std::move(netSrc)); }
    void operator=(NetMNISTIm2Col &netSrc) { NetBase::operator=(netSrc); ValueCopy(netSrc); }
    void operator=(NetMNISTIm2Col &&netSrc) { NetBase::operator=(std::move(netSrc)); ValueMove(std::move(netSrc)); }
private:
    // Meurons initialization for each layers
    void NeuronInit(dataset::MNIST &mnistTrainSet)
    {
        int iChannCnt = 1,
            iInputLnCnt = mnistTrainSet.ln_cnt(),
            iInputColCnt = mnistTrainSet.col_cnt();
        for(auto i=0ui64; i<seqLayer.size(); ++i) switch (seqLayer[i]->iLayerType)
        {
        case POOL_IM2COL:
            INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->NeuronInit(iInputLnCnt, iInputColCnt, iChannCnt, iNetBatchSize);
            iInputLnCnt = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->iLayerOutputLnCnt;
            iInputColCnt = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->iLayerOutputColCnt; break;
        case TRANS_IM2COL:
            if(INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->bFeatToVect)
            {
                INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iLnCnt = iInputLnCnt;
                INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iColCnt = iInputColCnt;
                iInputLnCnt = iInputLnCnt * iInputColCnt * iChannCnt;
                INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iElemCnt = iInputLnCnt;
                iInputColCnt = 1;
                iChannCnt = 1;
            }
            else
            {
                iInputLnCnt = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iLnCnt;
                iInputColCnt = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iColCnt;
                iChannCnt = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iElemCnt / (iInputLnCnt * iInputColCnt);
            } break;
        case FC:
            INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->NeuronInit(iInputLnCnt, iNetBatchSize); 
            iInputLnCnt = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->iLayerOutputLnCnt; break;
        case CONV_IM2COL:
            INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->NeuronInit(iInputLnCnt, iInputColCnt, iChannCnt, iNetBatchSize);
            iInputLnCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->iLayerOutputLnCnt;
            iInputColCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->iLayerOutputColCnt;
            iChannCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->iLayerKernelAmt; break;
        case PAD_CROP_IM2COL:
            INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->NeuronInit(iInputLnCnt, iInputColCnt);
            iInputLnCnt = INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->iLayerOutputLnCnt;
            iInputColCnt = INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->iLayerOutputColCnt; break;
        case CONV_BN_IM2COL:
            mapBNData.insert(i, BN_EXP_VAR());
            INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->NeuronInit(iChannCnt, iNetBatchSize); break;
        case FC_BN:
            mapBNData.insert(i, BN_EXP_VAR());
            INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->NeuronInit(iNetBatchSize); break;
        case ACT: INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->NeuronInit(iNetBatchSize); break;
        default: break;
        }
    }
    // For linear operation
    bool ForwProp(set<vect> &setInput)
    {
        for(auto i=0; i<seqLayer.size(); ++i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: setInput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->ForwProp(setInput); break;
            case POOL_IM2COL: setInput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->ForwProp(setInput); break;
            case TRANS_IM2COL: setInput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->ForwProp(setInput); break;
            case FC: setInput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->ForwProp(setInput); break;
            case FC_BN: setInput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->ForwProp(setInput); break;
            case CONV_IM2COL: setInput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->ForwProp(setInput); break;
            case CONV_BN_IM2COL: setInput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->ForwProp(setInput); break;
            case PAD_CROP_IM2COL: setInput = INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->ForwProp(setInput); break;
            default: return false;
            }
            if(!setInput.size()) return false;
        }
        return true;
    }
    bool BackProp(set<vect> &setOutput, set<vect> &setOrgn)
    {
        for(int i=seqLayer.size()-1; i>=0; --i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: setOutput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->BackProp(setOutput, setOrgn); break;
            case POOL_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            case TRANS_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            case FC: setOutput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->BackProp(setOutput); break;
            case FC_BN: setOutput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BackProp(setOutput); break;
            case CONV_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            case CONV_BN_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            case PAD_CROP_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->ForwProp(setOutput); break;
            default: return false;
            }
            if(!setOutput.size()) return false;
        }
        return true;
    }
    // For multi-thread batch operation
    bool ForwProp(vect &vecInput, uint64_t iIdx)
    {
        for(auto i=0; i<seqLayer.size(); ++i)
        {            
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: vecInput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            case POOL_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            case TRANS_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->ForwProp(vecInput); break;
            case FC: vecInput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            case CONV_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            case PAD_CROP_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->ForwProp(vecInput); break;
            case FC_BN:
                INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->setLayerInput[iIdx] = std::move(vecInput);
                // Detach all batch thread for main thread BN operation
                asyncConcurr.batch_thread_detach([this, i]{ iCurrLayerNo = i; iConcurrLayerTask = LAYER_OPT_FC_BN_FP; });
                // Attach all batch thread after the BN operation by main thread
                asyncConcurr.batch_thread_attach();
                vecInput = std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.setY[iIdx]); break;
            case CONV_BN_IM2COL:
                INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->setLayerInput[iIdx] = std::move(vecInput);
                // Detach all batch thread for main thread BN operation
                asyncConcurr.batch_thread_detach([this, i]{ iCurrLayerNo = i; iConcurrLayerTask = LAYER_OPT_CONV_BN_FP; });
                // Attach all batch thread after the BN operation by main thread
                asyncConcurr.batch_thread_attach();
                vecInput = std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.setIm2ColY[iIdx]); break;
            default: return false;
            }
            if(!vecInput.is_matrix()) return false;
        }
        return true;
    }
    bool BackProp(vect &vecOutput, vect &vecOrgn, uint64_t iIdx)
    {
        for(int i=seqLayer.size()-1; i>=0; --i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: vecOutput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->BackProp(vecOutput, iIdx, vecOrgn); break;
            case POOL_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case TRANS_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->BackProp(vecOutput); break;
            case FC: vecOutput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case CONV_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case PAD_CROP_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->ForwProp(vecOutput); break;
            case FC_BN:
                INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->setLayerOutputGrad[iIdx] = std::move(vecOutput);
                // Detach all batch thread for main thread BN operation
                asyncConcurr.batch_thread_detach([this, i]{ iCurrLayerNo = i; iConcurrLayerTask = LAYER_OPT_FC_BN_BP; });
                // Attach all batch thread after the BN operation by main thread
                asyncConcurr.batch_thread_attach();
                vecOutput = std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->setLayerInputGrad[iIdx]); break;
            case CONV_BN_IM2COL:
                INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->setLayerOutputGrad[iIdx] = std::move(vecOutput);
                // Detach all batch thread for main thread BN operation
                asyncConcurr.batch_thread_detach([this, i]{ iCurrLayerNo = i; iConcurrLayerTask = LAYER_OPT_CONV_BN_BP; });
                // Attach all batch thread after the BN operation by main thread
                asyncConcurr.batch_thread_attach();
                vecOutput = std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->setLayerInputGrad[iIdx]); break;
            default: return false;
            }
            if(!vecOutput.is_matrix()) return false;
        }
        return true;
    }
    // Update weight, kernel or hyper parameter
    void UpdatePara(uint64_t iCurrBatchIdx = 0, uint64_t iBatchCnt = 0)
    {
        auto bIsLastBatch = ((iCurrBatchIdx + 1) == iBatchCnt);
        for(auto i=0ui64; i<seqLayer.size(); ++i) switch (seqLayer[i] -> iLayerType)
        {
        case FC: INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->UpdatePara(); break;
        case FC_BN:
            INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                mapBNData[i].vecExp += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecMuBeta;
                mapBNData[i].vecVar += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecSigmaSqr;
                if(bIsLastBatch) _FC BNDeduceInit(mapBNData[i], iBatchCnt, iNetBatchSize);
            }
            else
            {
                mapBNData[i].vecExp = std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecMuBeta);
                mapBNData[i].vecVar = std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecSigmaSqr);
            } break;
        case CONV_IM2COL: INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->UpdatePara(); break;
        case CONV_BN_IM2COL:
            INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                mapBNData[i].vecExp += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColMuBeta;
                mapBNData[i].vecVar += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColSigmaSqr;
                if(bIsLastBatch) _CONV BNDeduceIm2ColInit(mapBNData[i], iBatchCnt, iNetBatchSize);
            }
            else
            {
                mapBNData[i].vecExp = std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData).vecIm2ColMuBeta;
                mapBNData[i].vecVar = std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData).vecIm2ColSigmaSqr;
            } break;
            default: continue;
        }
    }
    // Deduce test data
    bool Deduce(vect &vecInput)
    {
        for(auto i=0ui64; i<seqLayer.size(); ++i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: vecInput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->Deduce(vecInput); break;
            case POOL_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
            case TRANS_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
            case FC: vecInput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->Deduce(vecInput); break;
            case FC_BN: vecInput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->Deduce(vecInput, mapBNData[i]); break;
            case CONV_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
            case CONV_BN_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->Deduce(vecInput, mapBNData[i]); break;
            case PAD_CROP_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_PAD_CROP_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
            default: return false;
            }
            if(!vecInput.is_matrix()) return false;
        }
        return true;
    }
    bool RunLinear(dataset::MNIST &mnistTrainSet, dataset::MNIST &mnistTestSet)
    {
        mnistTrainSet.init_batch(iNetBatchSize);
        auto iEpoch = 0;
        double dAcc = 0, dPrec = 0, dRc = 0;
        do
        {
            CLOCK_BEGIN(0)
            mnistTrainSet.shuffle_batch(); ++ iEpoch;
            // Train
            for(auto i=0; i<mnistTrainSet.batch_cnt(); ++i)
            {
                CLOCK_BEGIN(1)
                mnistTrainSet.init_curr_set(i); dAcc = 0; dPrec = 0; dRc = 0;
                // FP
                if(!(ForwProp(mnistTrainSet.curr_elem_im2col) && deduce_acc_prec_rc(mnistTrainSet.curr_elem_im2col, mnistTrainSet.curr_lbl, dNetAcc, dAcc, dPrec, dRc))) return false;
                // EPOCH_TRAIN_STATUS(setOutput, mnistTrainSet.curr_orgn);
                // BP & update parameters
                if(BackProp(mnistTrainSet.curr_elem_im2col, mnistTrainSet.curr_orgn)) UpdatePara(i, mnistTrainSet.batch_cnt());
                else return false;
                dAcc /= mnistTrainSet.size();
                dPrec /= mnistTrainSet.size();
                dRc /= mnistTrainSet.size();
                CLOCK_END(1)
                EPOCH_TRAIN_STATUS(iEpoch, i+1, mnistTrainSet.batch_cnt(), dAcc, dPrec, dRc, CLOCK_DURATION(1));
            }
            // Deduce
            dAcc = 0; dPrec = 0; dRc = 0;
            mnistTestSet.init_curr_set();
            for(auto i=0; i<mnistTestSet.size(); ++i)
            {
                if(!(Deduce(mnistTestSet.curr_elem_im2col[i]) && deduce_acc_prec_rc(mnistTestSet.curr_elem_im2col, mnistTestSet.curr_lbl, dNetAcc, dAcc, dPrec, dRc))) return false;
                EPOCH_DEDUCE_PROG(i, mnistTestSet.size());
            }
            dAcc /= mnistTestSet.size();
            dPrec /= mnistTestSet.size();
            dRc /= mnistTestSet.size();
            CLOCK_END(0)
            EPOCH_DEDUCE_STATUS(iEpoch, dAcc, dPrec, dRc, CLOCK_DURATION(0));
            PRINT_ENTER
        } while (dRc < 1);
        return true;
    }
    bool RunThread(dataset::MNIST &mnistTrainSet, dataset::MNIST &mnistTestSet)
    {
        NetOutputDataBatch netTrainTemp;
        async::async_digit<bool> bComplete = false, // true - exit      false - continue
                                bException = false, // true - exception false - fine
                                bTrainMode = true;  // true - train     false - deduce
        // Batch thread        
        for(auto i=0; i<iNetBatchSize; ++i) asyncBatch.set_task(i, [this, &bComplete, &bException, &bTrainMode, &netTrainTemp, &mnistTrainSet, &mnistTestSet] (int idx)
        {
            while(true)
            {
                asyncConcurr.batch_thread_attach();
                if(bComplete || bException) break;
                if(bTrainMode)
                {
                    if(ForwProp(mnistTrainSet.curr_elem_im2col[idx], idx))
                    {
                        netTrainTemp.setBatchOutput[idx] = mnistTrainSet.curr_elem_im2col[idx];
                        if(!BackProp(mnistTrainSet.curr_elem_im2col[idx], mnistTrainSet.curr_orgn[idx], idx)) bException = true;
                    }
                    else bException = true;
                }
                else { if(!Deduce(mnistTestSet.curr_elem_im2col[idx])) bException = true; }
                asyncConcurr.batch_thread_detach([this]{ iConcurrLayerTask = LAYER_OPT_UPDATE_PARA; });
            }
        }, i);
        // Main thread
        async::async_queue<NetOutputDataBatch> setTrainOutput, setDeduceOutput;
        mnistTrainSet.init_batch(iNetBatchSize);
        mnistTestSet.init_batch(iNetBatchSize);
        auto iTrainSetBatchCnt = mnistTrainSet.batch_cnt(),
            iTestSetBatchCnt = mnistTestSet.batch_cnt();
        asyncPool.add_task([this, &iTrainSetBatchCnt, &iTestSetBatchCnt, &bComplete, &bException, &bTrainMode, &netTrainTemp, &mnistTrainSet, &mnistTestSet, &setTrainOutput, &setDeduceOutput]
        {
            do
            {
                mnistTrainSet.shuffle_batch();
                bTrainMode = true;
                for(auto i=0; i<iTrainSetBatchCnt; ++i)
                {
                    mnistTrainSet.init_curr_set(i);
                    netTrainTemp.setBatchOutput.init(iNetBatchSize);
                    do
                    {
                        asyncConcurr.main_thread_deploy_batch_thread();
                        if(bException)
                        {
                            asyncConcurr.main_thread_exception();
                            break;
                        }
                        switch (iConcurrLayerTask)
                        {
                        case LAYER_OPT_FC_BN_FP: INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerNo])->ForwPropCore(); break;
                        case LAYER_OPT_FC_BN_BP:  INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerNo])->BackPropCore(); break;
                        case LAYER_OPT_CONV_BN_FP: INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerNo])->ForwPropCore(); break;
                        case LAYER_OPT_CONV_BN_BP: INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerNo])->BackPropCore(); break;
                        case LAYER_OPT_UPDATE_PARA: UpdatePara(i, iTrainSetBatchCnt);
                        default: break;
                        }
                    } while(iConcurrLayerTask != LAYER_OPT_UPDATE_PARA);
                    if(bException) break;
                    netTrainTemp.setCurrOutputLbl = std::move(mnistTrainSet.curr_lbl);
                    setTrainOutput.en_queue(std::move(netTrainTemp));
                    asyncMainCtrl.thread_wake_one();
                }
                if(bException) break;
                bTrainMode = false;
                for(auto i=0; i<iTestSetBatchCnt; ++i)
                {
                    mnistTestSet.init_curr_set(i);
                    asyncConcurr.main_thread_deploy_batch_thread();
                    NetOutputDataBatch netTemp;
                    netTemp.setBatchOutput = std::move(mnistTestSet.curr_elem_im2col);
                    netTemp.setCurrOutputLbl = std::move(mnistTestSet.curr_lbl);
                    setDeduceOutput.en_queue(std::move(netTemp));
                    asyncMainCtrl.thread_wake_one();
                }
            } while (!bComplete);
        });
        // Date process
        double dAcc = 0, dPrec = 0, dRc = 0;
        auto iEpoch = 0;
        do
        {
            CLOCK_BEGIN(0)
            ++ iEpoch;
            // Train
            for(auto i=0; i<iTrainSetBatchCnt; ++i)
            {
                CLOCK_BEGIN(1)
                dAcc = 0; dPrec = 0; dRc = 0;
                if(!setTrainOutput.size()) asyncMainCtrl.thread_sleep();
                if(bException) return false;
                auto netCurrBatchIdxOutput = setTrainOutput.de_queue();
                deduce_acc_prec_rc(netCurrBatchIdxOutput.setBatchOutput, netCurrBatchIdxOutput.setCurrOutputLbl, dNetAcc, dAcc, dPrec, dRc); 
                CLOCK_END(1)
                EPOCH_TRAIN_STATUS(iEpoch, i+1, iTrainSetBatchCnt, dAcc, dPrec, dRc, CLOCK_DURATION(1));
            }
            // Deduce
            dAcc = 0; dPrec = 0; dRc = 0;
            for(auto i=0; i<iTestSetBatchCnt; ++i)
            {
                if(!setDeduceOutput.size()) asyncMainCtrl.thread_sleep();
                if(bException) return false;
                EPOCH_DEDUCE_PROG(i+1, iTestSetBatchCnt);
                auto netCurrBatchIdxOutput = setDeduceOutput.de_queue();
                deduce_acc_prec_rc(netCurrBatchIdxOutput.setBatchOutput, netCurrBatchIdxOutput.setCurrOutputLbl, dNetAcc, dAcc, dPrec, dRc, false);
            }
            dAcc /= mnistTestSet.size();
            dPrec /= mnistTestSet.size();
            dRc /= mnistTestSet.size();
            CLOCK_END(0)
            EPOCH_DEDUCE_STATUS(iEpoch, dAcc, dPrec, dRc, CLOCK_DURATION(0));
            PRINT_ENTER
            bComplete = (dRc == 1);
        } while (!bComplete);
        return true;
    }
public:
    NetMNISTIm2Col(double dNetAcc = 1e-5, uint64_t iBatchSize = 0, bool bUseMultiThread = true) : NetBase(dNetAcc, iBatchSize, bUseMultiThread) {}
    bool Run(dataset::MNIST &mnistTrainSet, dataset::MNIST &mnistTestSet)
    {
        NeuronInit(mnistTrainSet);
        if(bMultiThreadMode) return RunThread(mnistTrainSet, mnistTestSet);
        else return RunLinear(mnistTrainSet, mnistTestSet);
    }
    void Reset() { mapBNData.reset(); }
    ~NetMNISTIm2Col() { Reset(); }
private:
    NET_MAP<uint64_t, BN_EXP_VAR> mapBNData;
};

NEUNET_END