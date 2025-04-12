LAYER_BEGIN

struct Layer
{
    uint64_t iLayerType = ACT;
    double dLayerLearnRate = 0;

    virtual void ValueAssign(Layer &lyrSrc)
    {
        iLayerType = lyrSrc.iLayerType;
        dLayerLearnRate = lyrSrc.dLayerLearnRate;
    }
    virtual void ValueCopy(Layer &lyrSrc) { ValueAssign(lyrSrc); }
    virtual void ValueMove(Layer &&lyrSrc) {}
    Layer(Layer &lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(Layer &lyrSrc) { ValueCopy(lyrSrc); }
    
    Layer(uint64_t iLayerTypeVal = ACT, double dLearnRate = 0) : iLayerType(iLayerTypeVal), dLayerLearnRate(dLearnRate) {}
    void ForwProp() {}
    void BackProp() {}
    virtual void UpdatePara() {}

    virtual void Reset() {}
    ~Layer() {}
};

struct LayerAct : Layer
{
    uint64_t iLayerActFuncType = NULL;
    set<vect> setLayerInput;
    
    void ValueAssign(LayerAct &lyrSrc) { iLayerActFuncType = lyrSrc.iLayerActFuncType; }
    void ValueCopy(LayerAct &lyrSrc) { ValueAssign(lyrSrc); setLayerInput = lyrSrc.setLayerInput; }
    void ValueMove(LayerAct &&lyrSrc) { ValueAssign(lyrSrc); setLayerInput = std::move(lyrSrc.setLayerInput); }
    LayerAct(LayerAct &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerAct(LayerAct &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerAct &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerAct &&lyrSrc) { Layer::operator=(lyrSrc); ValueMove(std::move(lyrSrc)); }

    LayerAct(uint64_t iActFuncType) : Layer(ACT, 0), iLayerActFuncType(iActFuncType) {}
    void NeuronInit(uint64_t iBatchSize) { setLayerInput.init(iBatchSize); }

    vect ForwProp(vect &vecInput, uint64_t iIdx)
    {
        if(vecInput.is_matrix()) setLayerInput[iIdx] = std::move(vecInput);
        return Deduce(setLayerInput[iIdx]);
    }
    // Gradient is activation output for the last layer
    vect BackProp(vect &vecGrad, uint64_t iIdx, vect &vecOrgn)
    {
        switch (iLayerActFuncType)
        {
        case SIGMOID: return sigmoid_dv(setLayerInput[iIdx]).elem_cal_opt(vecGrad, MATRIX_ELEM_MULT);
        case RELU: return ReLU_dv(setLayerInput[iIdx]).elem_cal_opt(vecGrad, MATRIX_ELEM_MULT);
        case SOFTMAX: return softmax_cec_grad(vecGrad, vecOrgn);
        default: return vecGrad;
        }
    }

    set<vect> ForwProp(set<vect> &setInput)
    {
        set<vect> setAns(setInput.size());
        for(auto i=0; i<setAns.size(); ++i) setAns[i] = ForwProp(setInput[i], i);
        return setAns;
    }
    // Gradient is activation output for the last layer
    set<vect> BackProp(set<vect> &setGrad, set<vect> &setOrgn)
    {
        set<vect> setGradAns(setGrad.size());
        for(auto i=0; i<setGradAns.size(); ++i) setGradAns[i] = BackProp(setGrad[i], i, setOrgn[i]);
        return setGradAns;
    }
    vect Deduce(vect &vecInput)
    {
        switch (iLayerActFuncType)
        {
        case SIGMOID: return sigmoid(vecInput);
        case RELU: return ReLU(vecInput);
        case SOFTMAX: return softmax(vecInput);
        default: return vecInput;
        }
    }
    void Reset() { setLayerInput.reset(); }
    ~LayerAct() { Reset(); }
};

struct LayerFC : Layer
{
    uint64_t iLayerOutputLnCnt = 0;
    double dLayerWeightBoundryFirst = 0, dLayerWeightBoundrySecond = 0, dLayerWeightBoundryAcc = 0;
    vect vecLayerWeight, vecNesterovWeight;
    _ADA AdaDeltaVect advLayerDelta;
    _ADA AdaNesterovVect anvLayerMomt;
    set<vect> setLayerInput, setLayerGradWeight;

    void ValueAssign(LayerFC &lyrSrc)
    {
        iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt; dLayerWeightBoundryFirst = lyrSrc.dLayerWeightBoundryFirst; dLayerWeightBoundrySecond = lyrSrc.dLayerWeightBoundrySecond; dLayerWeightBoundryAcc = lyrSrc.dLayerWeightBoundryAcc;
    }
    void ValueCopy(LayerFC &lyrSrc)
    {
        ValueAssign(lyrSrc); vecLayerWeight = lyrSrc.vecLayerWeight; setLayerInput = lyrSrc.setLayerInput; anvLayerMomt = lyrSrc.anvLayerMomt; setLayerGradWeight = lyrSrc.setLayerGradWeight; vecNesterovWeight = lyrSrc.vecNesterovWeight; advLayerDelta = lyrSrc.advLayerDelta;
    }
    void ValueMove(LayerFC &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecLayerWeight = std::move(lyrSrc.vecLayerWeight); setLayerInput = std::move(lyrSrc.setLayerInput); anvLayerMomt = std::move(lyrSrc.anvLayerMomt); setLayerGradWeight = std::move(lyrSrc.setLayerGradWeight); vecNesterovWeight = std::move(lyrSrc.vecNesterovWeight); advLayerDelta = std::move(lyrSrc.advLayerDelta);
    }
    LayerFC(LayerFC &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFC(LayerFC &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFC &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFC &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }
    
    LayerFC(uint64_t iOutputLnCnt = 0, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5) : Layer(FC, dLearnRate), iLayerOutputLnCnt(iOutputLnCnt), dLayerWeightBoundryFirst(dRandBoundryFirst), dLayerWeightBoundrySecond(dRandBoundrySecond), dLayerWeightBoundryAcc(dRandBoundryAcc) {}
    void NeuronInit(uint64_t iInputLnCnt, uint64_t iBatchSize)
    {
        vecLayerWeight = _FC InitWeight(iInputLnCnt, iLayerOutputLnCnt, dLayerWeightBoundryFirst, dLayerWeightBoundrySecond, dLayerWeightBoundryAcc);
        vecNesterovWeight = vecLayerWeight;
        setLayerInput.init(iBatchSize);
        setLayerGradWeight.init(iBatchSize);
    }

    vect ForwProp(vect &vecInput, uint64_t iIdx)
    {
        if(vecInput.is_matrix()) setLayerInput[iIdx] = std::move(vecInput);
        if(dLayerLearnRate) return _FC Output(setLayerInput[iIdx], vecNesterovWeight);
        else return _FC Output(setLayerInput[iIdx], vecLayerWeight);
    }
    vect BackProp(vect &vecGrad, uint64_t iIdx)
    {
        setLayerGradWeight[iIdx] = _FC GradLossToWeight(vecGrad, setLayerInput[iIdx]);
        if(dLayerLearnRate) return _FC GradLossToInput(vecGrad, vecNesterovWeight);
        else return _FC GradLossToInput(vecGrad, vecLayerWeight);
    }

    set<vect> ForwProp(set<vect> &setInput)
    {
        set<vect> setAns(setInput.size());
        for(auto i=0; i<setAns.size(); ++i)
        {
            setAns[i] = ForwProp(setInput[i], i);
            if(!setAns[i].is_matrix()) return blank_vect_seq;
        }
        return setAns;
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        set<vect> setGradAns(setGrad.size());
        for(auto i=0; i<setGradAns.size(); ++i)
        {
            setGradAns[i] = BackProp(setGrad[i], i);
            if(!setGradAns[i].is_matrix()) return blank_vect_seq;
        }
        return setGradAns;
    }

    void UpdatePara()
    {
        auto vecLayerGradWeight = setLayerGradWeight.sum().elem_cal_opt(setLayerGradWeight.size(), MATRIX_ELEM_DIV);
        if(dLayerLearnRate)
        {
            vecLayerWeight = _FC AdaNesterovUpdateWeight(vecLayerWeight, vecLayerGradWeight, dLayerLearnRate, anvLayerMomt);
            vecNesterovWeight = anvLayerMomt.NesterovWeight(vecLayerWeight);
        }
        else vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, vecLayerGradWeight, advLayerDelta);
    }
    vect Deduce(vect &vecInput) { return _FC Output(vecInput, vecLayerWeight); }

    void ResetAda() { advLayerDelta.Reset(); anvLayerMomt.Reset(); }
    void Reset() { vecLayerWeight.reset(); setLayerInput.reset(); setLayerGradWeight.reset(); vecNesterovWeight.reset(); ResetAda(); }
    ~LayerFC() { Reset(); }
};

struct LayerFCBN : Layer
{
    // Shift, Scale, Dominant
    double dBeta = 0, dGamma = 1, dEpsilon = 1e-8, dGradBeta = 0, dGradGamma = 0, dNesterovBeta = dBeta, dNesterovGamma = dGamma;
    _ADA AdaDeltaVal advBeta, advGamma;
    _ADA AdaNesterovVal anvBeta, anvGamma;
    BN_FC BNData;
    set<vect> setLayerInput, setLayerOutputGrad, setLayerInputGrad;

    void ValueAssign(LayerFCBN &lyrSrc)
    {
        dBeta = lyrSrc.dBeta; dGamma = lyrSrc.dGamma; dNesterovBeta = lyrSrc.dNesterovBeta; dNesterovGamma = lyrSrc.dNesterovGamma; dEpsilon = lyrSrc.dEpsilon; dGradBeta = lyrSrc.dGradBeta; dGradGamma = lyrSrc.dGradGamma; advBeta = lyrSrc.advBeta; advGamma = lyrSrc.advGamma; anvBeta = lyrSrc.anvBeta; anvGamma = lyrSrc.anvGamma;
    }
    void ValueCopy(LayerFCBN &lyrSrc) { ValueAssign(lyrSrc); setLayerInput = lyrSrc.setLayerInput; setLayerInputGrad = lyrSrc.setLayerInputGrad; BNData = lyrSrc.BNData; setLayerOutputGrad = lyrSrc.setLayerOutputGrad; }
    void ValueMove(LayerFCBN &&lyrSrc) { ValueAssign(lyrSrc); setLayerInput = std::move(lyrSrc.setLayerInput); setLayerInputGrad = std::move(lyrSrc.setLayerInputGrad); BNData = std::move(lyrSrc.BNData); setLayerOutputGrad = std::move(lyrSrc.setLayerOutputGrad); }
    LayerFCBN(LayerFCBN &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFCBN(LayerFCBN &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFCBN &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFCBN &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerFCBN(double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-8) : Layer(FC_BN, dLearnRate), dBeta(dShift), dGamma(dScale), dNesterovBeta(dShift), dNesterovGamma(dScale), dEpsilon(dDmt) {}
    void NeuronInit(uint64_t iBatchSize) { setLayerInput.init(iBatchSize); setLayerOutputGrad.init(iBatchSize); }

    void ForwPropCore()
    {
        if(dLayerLearnRate) BNData = _FC BNTrain(setLayerInput, dNesterovBeta, dNesterovGamma, dEpsilon);
        else BNData = _FC BNTrain(setLayerInput, dBeta, dGamma, dEpsilon);
    }
    void BackPropCore()
    {
        dGradGamma = _FC BNGradLossToScale(setLayerOutputGrad, BNData);
        dGradBeta = _FC BNGradLossToShift(setLayerOutputGrad);
        if(dLayerLearnRate) setLayerInputGrad =  _FC BNGradLossToInput(BNData, setLayerInput, setLayerOutputGrad, dNesterovGamma, dEpsilon);
        else setLayerInputGrad = _FC BNGradLossToInput(BNData, setLayerInput, setLayerOutputGrad, dGamma, dEpsilon);
    }

    set<vect> ForwProp(set<vect> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        ForwPropCore();
        return std::move(BNData.setY);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        setLayerOutputGrad = std::move(setGrad);
        BackPropCore();
        return std::move(setLayerInputGrad);
    }
    void UpdatePara()
    {
        if(dLayerLearnRate)
        {
            dGamma = _FC BNAdaNesterovUpdateScaleShift(dGamma, dGradGamma, dLayerLearnRate,  anvGamma);
            dBeta = _FC BNAdaNesterovUpdateScaleShift(dBeta, dGradBeta, dLayerLearnRate, anvBeta);
            dNesterovBeta = anvBeta.NesterovWeight(dBeta);
            dNesterovGamma = anvBeta.NesterovWeight(dGamma);
        }
        else 
        {
            dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, dGradGamma, advGamma);
            dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, dGradBeta, advBeta);
        }
    }
    vect Deduce(vect &vecInput, BN_EXP_VAR &BNExpVar) { return _FC BNDeduce(vecInput, dBeta, dGamma, BNExpVar, dEpsilon); }

    void ResetAda() { advBeta.Reset(); advGamma.Reset(); anvBeta.Reset(); anvGamma.Reset(); }
    void Reset() { setLayerInput.reset(); setLayerOutputGrad.reset(); setLayerInputGrad.reset(); BNData.reset(); ResetAda(); }
    ~LayerFCBN() { Reset(); }
};

struct LayerConvIm2Col : Layer
{
    uint64_t iLayerInputLnCnt = 0, iLayerInputColCnt = 0, iLayerOutputLnCnt = 0, iLayerOutputColCnt = 0, iLayerKernelAmt = 0, iLayerKernelChannCnt = 0, iLayerKernelLnCnt = 0, iLayerKernelColCnt = 0 ,iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0;
    double dLayerKernelBoundryFirst = 0, dLayerKernelBoundrySecond = 0, dLayerKernelBoundryAcc = 0;
    vect vecKernel;
    _ADA AdaDeltaVect advKernel;
    _ADA AdaNesterovVect anvKernel;
    vect vecNesterovKernel;

    set<vect> setCaffeInput, setGradKernel;

    void ValueAssign(LayerConvIm2Col &lyrSrc)
    {
        iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt; iLayerInputColCnt = lyrSrc.iLayerInputColCnt; iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt; iLayerOutputColCnt = lyrSrc.iLayerOutputColCnt; iLayerKernelAmt = lyrSrc.iLayerKernelAmt; iLayerKernelChannCnt = lyrSrc.iLayerKernelChannCnt; iLayerKernelLnCnt = lyrSrc.iLayerKernelLnCnt; iLayerKernelColCnt = lyrSrc.iLayerKernelColCnt; iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; dLayerKernelBoundryFirst = lyrSrc.dLayerKernelBoundryFirst; dLayerKernelBoundrySecond = lyrSrc.dLayerKernelBoundrySecond; dLayerKernelBoundryAcc = lyrSrc.dLayerKernelBoundryAcc;
    }
    void ValueCopy(LayerConvIm2Col &lyrSrc)
    {
        ValueAssign(lyrSrc); vecKernel = lyrSrc.vecKernel; setCaffeInput = lyrSrc.setCaffeInput; advKernel = lyrSrc.advKernel; anvKernel = lyrSrc.anvKernel; setGradKernel = lyrSrc.setGradKernel; vecNesterovKernel = lyrSrc.vecNesterovKernel;
    }
    void ValueMove(LayerConvIm2Col &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecKernel = std::move(lyrSrc.vecKernel); setCaffeInput = std::move(lyrSrc.setCaffeInput); advKernel = std::move(lyrSrc.advKernel); anvKernel = std::move(lyrSrc.anvKernel); setGradKernel = std::move(lyrSrc.setGradKernel); vecNesterovKernel = std::move(lyrSrc.vecNesterovKernel);
    }
    LayerConvIm2Col(LayerConvIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConvIm2Col(LayerConvIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConvIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConvIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConvIm2Col(uint64_t iKernelAmt = 0, uint64_t iKernelLnCnt = 0, uint64_t iKernelColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, double dLearnRate = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5) : Layer(CONV_IM2COL, dLearnRate), iLayerKernelAmt(iKernelAmt), iLayerKernelLnCnt(iKernelLnCnt), iLayerKernelColCnt(iKernelColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), dLayerKernelBoundryFirst(dRandBoundryFirst), dLayerKernelBoundrySecond(dRandBoundrySecond), dLayerKernelBoundryAcc(dRandBoundryAcc) {}
    void NeuronInit(uint64_t iInputLnCnt, uint64_t iInputColCnt, uint64_t iInputChannCnt, uint64_t iBatchSize)
    {
        setCaffeInput.init(iBatchSize);
        setGradKernel.init(iBatchSize);
        vecKernel = _CONV InitKernelIm2Col(iLayerKernelAmt, iInputChannCnt, iLayerKernelLnCnt, iLayerKernelColCnt, dLayerKernelBoundryFirst, dLayerKernelBoundrySecond, dLayerKernelBoundryAcc);
        vecNesterovKernel = vecKernel;
        iLayerInputLnCnt = iInputLnCnt;
        iLayerInputColCnt = iInputColCnt;
        iLayerOutputLnCnt = SAMP_OUTPUT_DIR_CNT(iInputLnCnt, iLayerKernelLnCnt, iLayerLnStride, iLayerLnDilation);
        iLayerOutputColCnt = SAMP_OUTPUT_DIR_CNT(iInputColCnt, iLayerKernelColCnt, iLayerColStride, iLayerColDilation);
    }

    vect ForwProp(vect &vecInput, uint64_t iIdx)
    {
        if(vecInput.is_matrix()) setCaffeInput[iIdx] = _CONV Im2ColInputCaffeTransform(vecInput, iLayerInputLnCnt, iLayerInputColCnt, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
        if(dLayerLearnRate) return _CONV ConvIm2Col(setCaffeInput[iIdx], vecNesterovKernel);
        else return _CONV ConvIm2Col(setCaffeInput[iIdx], vecKernel);
    }
    vect BackProp(vect &vecGrad, uint64_t iIdx)
    {
        setGradKernel[iIdx] = _CONV GradLossToKernelIm2Col(vecGrad, setCaffeInput[iIdx]);
        if(dLayerLearnRate) vecGrad = _CONV GradLossToConvIm2ColCaffeInput(vecGrad, vecNesterovKernel);
        else vecGrad = _CONV GradLossToConvIm2ColCaffeInput(vecGrad, vecKernel);
        return _CONV Im2ColInputCaffeTransform(vecGrad, iLayerInputLnCnt, iLayerInputColCnt, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, true);
    }

    set<vect> ForwProp(set<vect> &setInput)
    {
        set<vect> setAns(setInput.size());
        for(auto i=0; i<setAns.size(); ++i)
        {
            setAns[i] = ForwProp(setInput[i], i);
            if(!setAns[i].is_matrix()) return blank_vect_seq;
        }
        return setAns;
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        set<vect> setGradAns(setGrad.size());
        for(auto i=0; i<setGradAns.size(); ++i)
        {
            setGradAns[i] = BackProp(setGrad[i], i);
            if(!setGradAns[i].is_matrix()) return blank_vect_seq;
        }
        return setGradAns;
    }
    void UpdatePara()
    {
        auto vecGradKernel = setGradKernel.sum().elem_cal_opt(setGradKernel.size(), MATRIX_ELEM_DIV);
        if(dLayerLearnRate) 
        {
            vecKernel = _FC AdaNesterovUpdateWeight(vecKernel, vecGradKernel, dLayerLearnRate, anvKernel);
            vecNesterovKernel = anvKernel.NesterovWeight(vecKernel);
        }
        else vecKernel = _FC AdaDeltaUpdateWeight(vecKernel, vecGradKernel, advKernel);
    }
    vect Deduce(vect &vecInput) { return _CONV ConvIm2Col(_CONV Im2ColInputCaffeTransform(vecInput, iLayerInputLnCnt, iLayerInputColCnt, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation), vecKernel); }

    void ResetAda() { advKernel.Reset(); anvKernel.Reset(); }
    void Reset() { setCaffeInput.reset(); vecKernel.reset(); vecNesterovKernel.reset(); ResetAda(); }
    ~LayerConvIm2Col() { Reset(); }
};

struct LayerConvBNIm2Col : Layer
{
    // Dominant
    double dBeta = 0, dGamma = 1, dEpsilon = 1e-8;
    // Shift, Scale
    vect vecBeta, vecGamma, vecGradBeta, vecGradGamma, vecNesterovBeta, vecNesterovGamma;
    _ADA AdaDeltaVect advBeta, advGamma;
    _ADA AdaNesterovVect anvBeta, anvGamma;
    BN_CONV_IM2COL BNData;

    set<vect> setLayerInput, setLayerOutputGrad, setLayerInputGrad;

    void ValueAssign(LayerConvBNIm2Col &lyrSrc) { dBeta = lyrSrc.dBeta; dGamma = lyrSrc.dGamma; dEpsilon = lyrSrc.dEpsilon; }
    void ValueCopy(LayerConvBNIm2Col &lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = lyrSrc.vecBeta; vecGamma = lyrSrc.vecGamma; setLayerInput = lyrSrc.setLayerInput; setLayerOutputGrad = lyrSrc.setLayerOutputGrad; setLayerInputGrad = lyrSrc.setLayerInputGrad; advBeta = lyrSrc.advBeta; advGamma = lyrSrc.advGamma; anvBeta = lyrSrc.anvBeta; anvGamma = lyrSrc.anvGamma; BNData = lyrSrc.BNData; vecGradBeta = lyrSrc.vecGradBeta; vecGradGamma = lyrSrc.vecGradGamma; vecNesterovBeta = lyrSrc.vecNesterovBeta; vecNesterovGamma = lyrSrc.vecNesterovGamma;
    }
    void ValueMove(LayerConvBNIm2Col &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = std::move(lyrSrc.vecBeta); vecGamma = std::move(lyrSrc.vecGamma); setLayerInput = std::move(lyrSrc.setLayerInput); setLayerOutputGrad = std::move(lyrSrc.setLayerOutputGrad); setLayerInputGrad = std::move(lyrSrc.setLayerInputGrad); advBeta = std::move(lyrSrc.advBeta); advGamma = std::move(lyrSrc.advGamma); anvBeta = std::move(lyrSrc.anvBeta); anvGamma = std::move(lyrSrc.anvGamma); BNData = std::move(lyrSrc.BNData); vecGradBeta = std::move(lyrSrc.vecGradBeta); vecGradGamma = std::move(lyrSrc.vecGradGamma); vecNesterovBeta = std::move(lyrSrc.vecNesterovBeta); vecNesterovGamma = std::move(lyrSrc.vecNesterovGamma);
    }
    LayerConvBNIm2Col(LayerConvBNIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConvBNIm2Col(LayerConvBNIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConvBNIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConvBNIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConvBNIm2Col(double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-8) : Layer(CONV_BN_IM2COL, dLearnRate), dBeta(dShift), dGamma(dScale), dEpsilon(dDmt) {}
    void NeuronInit(uint64_t iInputChannCnt, uint64_t iBatchSize)
    {
        vecBeta = _CONV BNInitScaleShift(iInputChannCnt, dBeta);
        vecGamma = _CONV BNInitScaleShift(iInputChannCnt, dGamma);
        vecNesterovBeta = vecBeta; vecNesterovGamma = vecGamma;
        setLayerInput.init(iBatchSize); setLayerOutputGrad.init(iBatchSize);
    }

    void ForwPropCore()
    {
        if(dLayerLearnRate) BNData = _CONV BNTrainIm2Col(setLayerInput, vecNesterovBeta, vecNesterovGamma, dEpsilon);
        else BNData = _CONV BNTrainIm2Col(setLayerInput, vecBeta, vecGamma, dEpsilon);
    }
    void BackPropCore()
    {
        vecGradGamma = _CONV BNGradLossToScaleIm2Col(setLayerOutputGrad, BNData);
        vecGradBeta = _CONV BNGradLossToShiftIm2Col(setLayerOutputGrad);
        if(dLayerLearnRate) setLayerInputGrad = _CONV BNGradLossToInputIm2Col(setLayerOutputGrad, BNData, setLayerInput, vecNesterovGamma, dEpsilon);
        else setLayerInputGrad = _CONV BNGradLossToInputIm2Col(setLayerOutputGrad, BNData, setLayerInput, vecGamma, dEpsilon);
    }
    
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        ForwPropCore();
        return std::move(BNData.setIm2ColY);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        setLayerOutputGrad = std::move(setGrad);
        BackPropCore();
        return std::move(setLayerInputGrad);
    }
    void UpdatePara()
    {
        if(dLayerLearnRate)
        {
            vecGamma = _CONV BNAdaNesterovUpdateScaleShift(vecGamma, vecGradGamma, dLayerLearnRate, anvGamma);
            vecBeta = _CONV BNAdaNesterovUpdateScaleShift(vecBeta, vecGradBeta, dLayerLearnRate, anvBeta);
            vecNesterovBeta = anvBeta.NesterovWeight(vecBeta);
            vecNesterovGamma = anvBeta.NesterovWeight(vecGamma);
        }
        else 
        {
            vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, vecGradGamma, advGamma);
            vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, vecGradBeta, advBeta);
        }
    }
    vect Deduce(vect &vecInput, BN_EXP_VAR &BNExpVar) { return _CONV BNDeduceIm2Col(vecInput, vecBeta, vecGamma, BNExpVar, dEpsilon); }

    void ResetAda() { advBeta.Reset(); advGamma.Reset(); anvBeta.Reset(); anvGamma.Reset(); }
    void Reset() { vecBeta.reset(); vecGamma.reset(); setLayerInput.reset(); setLayerOutputGrad.reset(); setLayerInputGrad.reset(); vecGradBeta.reset(); vecGradGamma.reset(); BNData.Reset(); vecNesterovBeta.reset(); vecNesterovGamma.reset(); ResetAda(); }
    ~LayerConvBNIm2Col() { Reset(); }
};

struct LayerPoolIm2Col : Layer
{
    uint64_t iPoolType = POOL_MAX_IM2COL, iLayerInputLnCnt = 0, iLayerInputColCnt = 0, iLayerOutputLnCnt = 0, iLayerOutputColCnt = 0, iLayerFilterLnCnt = 0, iLayerFilterColCnt = 0, iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0;

    vect_t<bagrt::net_list<mtx::mtx_pos>> setInputMaxPosList/* Max */;

    void ValueAssign(LayerPoolIm2Col &lyrSrc)
    {
        iPoolType = lyrSrc.iPoolType; iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt; iLayerOutputColCnt = lyrSrc.iLayerOutputColCnt; iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt; iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt; iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt; iLayerInputColCnt = lyrSrc.iLayerInputColCnt;
    }
    void ValueCopy(LayerPoolIm2Col &lyrSrc) { ValueAssign(lyrSrc); setInputMaxPosList = lyrSrc.setInputMaxPosList; }
    void ValueMove(LayerPoolIm2Col &&lyrSrc) { ValueAssign(lyrSrc); setInputMaxPosList = std::move(lyrSrc.setInputMaxPosList); }
    LayerPoolIm2Col(LayerPoolIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerPoolIm2Col(LayerPoolIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerPoolIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerPoolIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerPoolIm2Col(uint64_t iPoolTypeVal = POOL_MAX_IM2COL, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0) : Layer(POOL_IM2COL, 0), iPoolType(iPoolTypeVal), iLayerFilterLnCnt(iFilterLnCnt), iLayerFilterColCnt(iFilterColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation) {}
    void NeuronInit(uint64_t iInputLnCnt, uint64_t iInputColCnt, uint64_t iInputChannCnt, uint64_t iBatchSize)
    {
        iLayerInputLnCnt = iInputLnCnt;
        iLayerInputColCnt = iInputColCnt;
        auto iPoolMaxInputElemCnt = 0;
        if(iPoolType == POOL_GAG_IM2COL) { iLayerOutputLnCnt = 1; iLayerOutputColCnt = 1; }
        else
        {
            iLayerOutputLnCnt = SAMP_OUTPUT_DIR_CNT(iInputLnCnt, iLayerFilterLnCnt, iLayerLnStride, iLayerLnDilation);
            iLayerOutputColCnt = SAMP_OUTPUT_DIR_CNT(iInputColCnt, iLayerFilterColCnt, iLayerColStride, iLayerColDilation);
            if(iPoolType == POOL_MAX_IM2COL)
            {
                auto iPoolMaxOutputElemCnt = iLayerOutputLnCnt * iLayerOutputColCnt * iInputChannCnt;
                setInputMaxPosList.init(iBatchSize);
                for(auto i=0; i<iBatchSize; ++i) setInputMaxPosList[i].init(iPoolMaxOutputElemCnt);
            }
        }
    }

    vect ForwProp(vect &vecInput, uint64_t iIdx)
    {
        if(iPoolType == POOL_GAG_IM2COL) return _CONV PoolGlbAvgIm2Col(vecInput);
        else return _CONV PoolMaxAvgIm2Col(iPoolType, _CONV Im2ColInputCaffeTransform(vecInput, iLayerInputLnCnt, iLayerInputColCnt, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation), setInputMaxPosList[iIdx], iLayerFilterLnCnt, iLayerFilterColCnt);
    }
    vect BackProp(vect &vecGrad, uint64_t iIdx)
    {
        if(iPoolType == POOL_GAG_IM2COL) return _CONV GradLossToPoolGlbAvgInputIm2Col(vecGrad, iLayerInputLnCnt*iLayerInputColCnt);
        else return _CONV Im2ColInputCaffeTransform(_CONV GradLossToPoolMaxAvgIm2ColCaffeInput(iPoolType, vecGrad, setInputMaxPosList[iIdx], iLayerFilterLnCnt, iLayerFilterColCnt), iLayerInputLnCnt, iLayerInputColCnt, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, true);
    }

    set<vect> ForwProp(set<vect> &setInput)
    {
        set<vect> setAns(setInput.size());
        for(auto i=0; i<setAns.size(); ++i) setAns[i] = ForwProp(setInput[i], i);
        return setAns;
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        set<vect> setGradAns(setGrad.size());
        for(auto i=0; i<setGradAns.size(); ++i) setGradAns[i] = BackProp(setGrad[i], i);
        return setGradAns;
    }
    vect Deduce(vect &vecInput)
    {
        if(iPoolType == POOL_GAG_IM2COL) return _CONV PoolGlbAvgIm2Col(vecInput);
        else return _CONV PoolMaxAvgIm2Col(iPoolType, _CONV Im2ColInputCaffeTransform(vecInput, iLayerInputLnCnt, iLayerInputColCnt, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation), set<bagrt::net_list<mtx::mtx_pos>>(), iLayerFilterLnCnt, iLayerFilterColCnt);
    }

    void Reset() { setInputMaxPosList.reset(); }
    ~LayerPoolIm2Col() { Reset(); }
};

struct LayerPadCropIm2Col : Layer
{
    bool bLayerPadMode = true;
    uint64_t iLayerInputLnCnt = 0, iLayerInputColCnt = 0, iLayerOutputLnCnt = 0, iLayerOutputColCnt = 0, iLayerAlterTop = 0, iLayerAlterRight = 0, iLayerAlterBottom = 0, iLayerAlterLeft = 0, iLayerAlterLnDistance = 0, iLayerAlterColDistance = 0;
    void ValueAssign(LayerPadCropIm2Col &lyrSrc)
    {
        bLayerPadMode = lyrSrc.bLayerPadMode; iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt, iLayerInputColCnt = lyrSrc.iLayerInputColCnt, iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt, iLayerOutputColCnt = lyrSrc.iLayerOutputColCnt, iLayerAlterTop = lyrSrc.iLayerAlterTop; iLayerAlterRight = lyrSrc.iLayerAlterRight; iLayerAlterBottom = lyrSrc.iLayerAlterBottom; iLayerAlterLeft = lyrSrc.iLayerAlterLeft; iLayerAlterLnDistance = lyrSrc.iLayerAlterLnDistance; iLayerAlterColDistance = lyrSrc.iLayerAlterColDistance;
    }
    void ValueCopy(LayerPadCropIm2Col &lyrSrc) { ValueAssign(lyrSrc); }
    LayerPadCropIm2Col(LayerPadCropIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(LayerPadCropIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }

    LayerPadCropIm2Col(bool bPadMode = true, uint64_t iAlterTop = 0, uint64_t iAlterRight = 0, uint64_t iAlterBottom = 0, uint64_t iAlterLeft = 0, uint64_t iAlterLnDistance = 0, uint64_t iAlterColDistance = 0) : Layer(PAD_CROP_IM2COL, 0), bLayerPadMode(bPadMode), iLayerAlterTop(iAlterTop), iLayerAlterRight(iAlterRight), iLayerAlterBottom(iAlterBottom), iLayerAlterLeft(iAlterLeft), iLayerAlterLnDistance(iAlterLnDistance), iLayerAlterColDistance(iAlterColDistance) {}
    void NeuronInit(uint64_t iInputLnCnt, uint64_t iInputColCnt)
    {
        iLayerInputLnCnt = iInputLnCnt;
        iLayerInputColCnt = iInputColCnt;
        if(bLayerPadMode)
        {
            iLayerOutputLnCnt = mtx::mtx_pad_cnt(iLayerAlterTop, iLayerAlterBottom, iInputLnCnt, iLayerAlterLnDistance);
            iLayerOutputColCnt = mtx::mtx_pad_cnt(iLayerAlterLeft, iLayerAlterRight, iInputColCnt, iLayerAlterColDistance);
        }
        else
        {
            iLayerOutputLnCnt = mtx::mtx_crop_cnt(iLayerAlterTop, iLayerAlterBottom, iInputLnCnt, iLayerAlterLnDistance);
            iLayerOutputColCnt = mtx::mtx_crop_cnt(iLayerAlterLeft, iLayerAlterRight, iInputColCnt, iLayerAlterColDistance);
        }
    }

    // [bPadMode] true - pad; false - crop
    vect Im2ColPadCrop(vect &vecSrc, bool bPadMode)
    {
        if(bPadMode) return _CONV Im2ColFeaturePad(vecSrc, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerInputLnCnt, iLayerInputColCnt, iLayerAlterTop, iLayerAlterRight, iLayerAlterBottom, iLayerAlterLeft, iLayerAlterLnDistance, iLayerAlterColDistance);
        else return _CONV Im2ColFeatureCrop(vecSrc, iLayerOutputLnCnt, iLayerOutputColCnt, iLayerInputLnCnt, iLayerInputColCnt, iLayerAlterTop, iLayerAlterRight, iLayerAlterBottom, iLayerAlterLeft, iLayerAlterLnDistance, iLayerAlterColDistance);
    }

    vect ForwProp(vect &vecSrc) { return Deduce(vecSrc); }
    vect BackProp(vect &vecGrad) { return Im2ColPadCrop(vecGrad, !bLayerPadMode); }

    set<vect> ForwProp(set<vect> &setSrc)
    {
        set<vect> setAns(setSrc.size());
        for(auto i=0; i<setAns.size(); ++i) setAns[i] = ForwProp(setSrc[i]);
        return setAns;
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        set<vect> setGradAns(setGrad.size());
        for(auto i=0; i<setGradAns.size(); ++i) setGradAns[i] = BackProp(setGrad[i]);
        return setGradAns;
    }

    vect Deduce(vect &vecSrc) { return Im2ColPadCrop(vecSrc, bLayerPadMode); }
};

struct LayerTransIm2Col : Layer
{
    bool bFeatToVect = true;
    uint64_t iLnCnt = 0, iColCnt = 0, iElemCnt = 0;

    void ValueAssign(LayerTransIm2Col &lyrSrc) { bFeatToVect = lyrSrc.bFeatToVect; iLnCnt = lyrSrc.iLnCnt; iColCnt = lyrSrc.iColCnt; iElemCnt = lyrSrc.iElemCnt; }
    void ValueCopy(LayerTransIm2Col &lyrSrc) { ValueAssign(lyrSrc); }
    LayerTransIm2Col(LayerTransIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(LayerTransIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }

    LayerTransIm2Col(uint64_t iChannLnCnt = 0, uint64_t iChannColCnt = 0, uint64_t iChannCnt = 1) : Layer(TRANS_IM2COL, 0), iLnCnt(iChannLnCnt), iColCnt(iChannColCnt)
    {
        if(iChannLnCnt && iChannColCnt)
        {
            bFeatToVect = false;
            iElemCnt = iLnCnt * iColCnt * iChannCnt;
        }
        else if(!(iChannLnCnt || iChannColCnt) && iChannCnt==1) bFeatToVect = true;
        else throw std::logic_error("Mistake syntax for line & column count of each channel.");
    }
    
    // [bFlat] true - feature to vector; false - vector to feature
    vect Im2ColTrans(vect &vecSrc, bool bFlat)
    {
        if(bFlat) return _FC FeatureTransformIm2Col(vecSrc);
        else
        {
            if(vecSrc.COL_CNT==1 && iElemCnt==vecSrc.ELEM_CNT) return _FC FeatureTransformIm2Col(vecSrc, iLnCnt, iColCnt);
            else return blank_vect;
        }
    }

    vect ForwProp(vect &vecInput) { return Deduce(vecInput); }
    vect BackProp(vect &vecGrad) { return Im2ColTrans(vecGrad, !bFeatToVect); }

    set<vect> ForwProp(set<vect> &setInput)
    {
        set<vect> setAns(setInput.size());
        for(auto i=0; i<setAns.size(); ++i) setAns[i] = ForwProp(setInput[i]);
        return setAns;
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        set<vect> setGradAns(setGrad.size());
        for(auto i=0; i<setGradAns.size(); ++i) setGradAns[i] = BackProp(setGrad[i]);
        return setGradAns;
    }

    vect Deduce(vect &vecInput) { return Im2ColTrans(vecInput, bFeatToVect); }
};

LAYER_END