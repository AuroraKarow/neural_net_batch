CONV_BEGIN

tensor InitKernel(uint64_t iAmt, uint64_t iChannCnt, uint64_t iLnCnt, uint64_t iColCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    tensor tenKernel(iAmt);
    for(auto i=0; i<iAmt; ++i)
    {
        tenKernel[i].init(iChannCnt);
        for(auto j=0; j<iChannCnt; ++j) tenKernel[i][j] = vect(iLnCnt, iColCnt, true, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    }
    return tenKernel;
}

set<feature> Conv(set<feature> &setInput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    set<feature> setOutput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOutput[i] = Conv(setInput[i], tenKernel, iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(!setOutput[i].size()) return blank_ft_seq;
    }
    return setOutput;
}

tensor GradLossToKernel(set<feature> &setGradLossToOutput, set<feature> &setInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    tensor tenGradLossToKernel;
    for(auto i=0; i<setInput.size(); ++i)
    {
        auto tenSglGrad = GradLossToKernel(setGradLossToOutput[i], setInput[i], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(tenGradLossToKernel.size()) for(auto j=0; j<tenSglGrad.size(); ++j) for(auto k=0; k<tenSglGrad[i].size(); ++k) tenGradLossToKernel[j][k] += tenSglGrad[j][k];
        else tenGradLossToKernel = std::move(tenSglGrad);
        if(!tenGradLossToKernel.size()) return blank_tensor;
    }
    return tenGradLossToKernel;
}

set<feature> GradLossToInput(set<feature> &setGradLossToOutput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    set<feature> setGradLossToInput(setGradLossToOutput.size());
    for(auto i=0; i<setGradLossToOutput.size(); ++i)
    {
        setGradLossToInput[i] = GradLossToInput(setGradLossToOutput[i], tenKernel, iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(!setGradLossToInput[i].size()) return blank_ft_seq;
    }
    return setGradLossToInput;
}

set<feature> Pool(set<feature> &vecInput, uint64_t iPoolType = POOL_DOWN_MAX, bool bDownSamp = true, set<feature> &setTraceInput = set<feature>(), uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    set<feature> setOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i)
    {
        if(bDownSamp) setOutput[i] = PoolDown(vecInput[i], iPoolType, iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);
        else setOutput[i] = PoolUp(vecInput[i], iPoolType, setTraceInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);
        if(!setOutput[i].size()) return blank_ft_seq;
    }
    return setOutput;
}

struct ConvBN
{
    feature vecMiuBeta;
    feature vecSigmaSqr;
    set<feature> setBarX;
    set<feature> setY;
    ConvBN(){}
    void ValueCopy(ConvBN &ConvBNVal)
    {
        vecMiuBeta = ConvBNVal.vecMiuBeta;
        vecSigmaSqr = ConvBNVal.vecSigmaSqr;
        setBarX = ConvBNVal.setBarX;
        setY = ConvBNVal.setY;
    }
    void ValueMove(ConvBN &&ConvBNVal)
    {
        vecMiuBeta = std::move(ConvBNVal.vecMiuBeta);
        vecSigmaSqr = std::move(ConvBNVal.vecSigmaSqr);
        setBarX = std::move(ConvBNVal.setBarX);
        setY = std::move(ConvBNVal.setY);
    }
    ConvBN(ConvBN &ConvBNVal) { ValueCopy(ConvBNVal); }
    ConvBN(ConvBN &&ConvBNVal) { ValueMove(std::move(ConvBNVal)); }
    void operator=(ConvBN &ConvBNVal) { ValueCopy(ConvBNVal); }
    void operator=(ConvBN &&ConvBNVal) { ValueMove(std::move(ConvBNVal)); }
    void Reset()
    {
        vecMiuBeta.reset();
        vecSigmaSqr.reset();
        setBarX.reset();
        setY.reset();
    }
    ~ConvBN() { Reset(); }
};

vect BNInitScaleShift(uint64_t iChannCnt, double dFillVal)
{
    vect vecSS(iChannCnt, IDX_SGL);
    if(dFillVal) for(auto i=0; i<iChannCnt; ++i) vecSS.pos_idx(i) = dFillVal;
    return vecSS;
}

ConvBN BNTrain(set<feature> &setInput, vect &vecBeta, vect &vecGamma, double dEpsilon = 1e-8)
{
    ConvBN BNOutput;
    // Average & Variance
    BNOutput.vecMiuBeta.init(setInput[IDX_ZERO].size());
    BNOutput.vecSigmaSqr.init(setInput[IDX_ZERO].size());
    for(auto i=0; i<setInput[IDX_ZERO].size(); ++i)
    {
        for(auto j=0; j<setInput.size(); ++j)
            if(BNOutput.vecMiuBeta[i].is_matrix()) BNOutput.vecMiuBeta[i] += setInput[j][i];
            else BNOutput.vecMiuBeta[i] = setInput[j][i];
        BNOutput.vecMiuBeta[i] = BNOutput.vecMiuBeta[i].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
        for(auto j=0; j<setInput.size(); ++j)
        {
            auto vecSglSigmaSqr = (setInput[j][i] - BNOutput.vecMiuBeta[i]).elem_cal_opt(2, MATRIX_ELEM_POW);
            if(BNOutput.vecSigmaSqr[i].is_matrix()) BNOutput.vecSigmaSqr[i] += vecSglSigmaSqr;
            else BNOutput.vecSigmaSqr[i] = std::move(vecSglSigmaSqr);
        }
        BNOutput.vecSigmaSqr[i] = BNOutput.vecSigmaSqr[i].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    }
    // Normalize & Output
    BNOutput.setBarX.init(setInput.size());
    BNOutput.setY.init(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        BNOutput.setBarX[i].init(setInput[i].size());
        for(auto j=0; j<setInput[i].size(); ++j) BNOutput.setBarX[i][j] = (setInput[i][j] - BNOutput.vecMiuBeta[j]).elem_cal_opt(DIV_DOM(BNOutput.vecSigmaSqr[j], dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
        BNOutput.setY[i].init(setInput[i].size());
        for(auto j=0; j<setInput[i].size(); ++j) BNOutput.setY[i][j] = (vecGamma.pos_idx(j) * BNOutput.setBarX[i][j]).broadcast_add(vecBeta.pos_idx(j));
    }
    return BNOutput;
}

set<feature> BNGradLossToInput(ConvBN &ConvBNOutput, set<feature> &setInput, set<feature> &setGradLossToOutput, vect &vecGamma, double dEpsilon = 1e-8)
{
    // Sigma & Square-powered sigma
    feature vecDmrSigmaSqr(ConvBNOutput.vecSigmaSqr.size()), vecDmrSigma(ConvBNOutput.vecSigmaSqr.size());
    for(auto i=0; i<ConvBNOutput.vecSigmaSqr.size(); ++i)
    {
        vecDmrSigmaSqr[i] = DIV_DOM(ConvBNOutput.vecSigmaSqr[i], dEpsilon);
        vecDmrSigma[i] = vecDmrSigmaSqr[i].elem_cal_opt(0.5, MATRIX_ELEM_POW);
    }
    // Gradient loss to normalized output
    set<feature> setGradLossToBarX(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setGradLossToBarX[i].init(setInput[i].size());
        for(auto j=0; j<ConvBNOutput.vecMiuBeta.size(); ++j) setGradLossToBarX[i][j] = setGradLossToOutput[i][j] * vecGamma.pos_idx(j);
    }
    // Gradient loss to square-powered sigma
    feature vecGradLossToSigmaSqr(ConvBNOutput.vecSigmaSqr.size());
    for(auto i=0; i<ConvBNOutput.vecSigmaSqr.size(); ++i) for(auto j=0; j<setInput.size(); ++j)
    {
        auto vecSglGradLossToSigmaSqr = ((-1) * setGradLossToBarX[j][i].elem_cal_opt((setInput[j][i] - ConvBNOutput.vecMiuBeta[i]), MATRIX_ELEM_MULT)).elem_cal_opt((2 * vecDmrSigmaSqr[i].elem_cal_opt(1.5, MATRIX_ELEM_POW)), MATRIX_ELEM_DIV);
        if(vecGradLossToSigmaSqr[i].is_matrix()) vecGradLossToSigmaSqr[i] += vecSglGradLossToSigmaSqr;
        else vecGradLossToSigmaSqr[i] = std::move(vecSglGradLossToSigmaSqr);
    }
    // Gradient loss to miubeta
    feature vecGradLossToMiuBeta(ConvBNOutput.vecMiuBeta.size());
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i)
    {
        vect vecDistribute;
        vect vecDistance;
        for(auto j=0; j<setInput.size(); ++j)
        {
            auto vecSglDistribute = (-1) * setGradLossToBarX[j][i].elem_cal_opt(vecDmrSigma[i], MATRIX_ELEM_DIV);
            if(vecDistribute.is_matrix()) vecDistribute += vecSglDistribute;
            else vecDistribute = vecSglDistribute;
            auto vecSglDistance = (-2) * (setInput[j][i] - ConvBNOutput.vecMiuBeta[i]);
            if(vecDistance.is_matrix()) vecDistance += vecSglDistance;
            else vecDistance = vecSglDistance;
        }
        vecGradLossToMiuBeta[i] = vecDistribute + vecGradLossToSigmaSqr[i].elem_cal_opt(vecDistance.elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV), MATRIX_ELEM_MULT);
    }
    // Gradient loss to input
    set<feature> setGradLossToInput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setGradLossToInput[i].init(setInput[i].size());
        for(auto j=0; j<setInput[i].size(); ++j) setGradLossToInput[i][j] = setGradLossToBarX[i][j].elem_cal_opt(vecDmrSigma[j], MATRIX_ELEM_DIV) + (2 / setInput.size()) * vecGradLossToSigmaSqr[j].elem_cal_opt((setInput[i][j] - ConvBNOutput.vecMiuBeta[j]), MATRIX_ELEM_MULT) + vecGradLossToMiuBeta[j].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    }
    return setGradLossToInput;
}

vect BNGradLossToScale(set<feature> &setGradLossToOutput, ConvBN &ConvBNOutput)
{
    vect vecGradGamma(ConvBNOutput.vecMiuBeta.size(), IDX_SGL);
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i)
        for(auto j=0; j<setGradLossToOutput.size(); ++j)
            for(auto k=0; k<setGradLossToOutput[j][i].ELEM_CNT; ++k)
                vecGradGamma.pos_idx(i) += setGradLossToOutput[j][i].pos_idx(k) * ConvBNOutput.setBarX[j][i].pos_idx(k);
    return vecGradGamma;
}

vect BNGradLossToShift(set<feature> &setGradLossToOutput)
{
    vect vecGradBeta(setGradLossToOutput[IDX_ZERO].size(), IDX_SGL);
    for(auto i=0; i<setGradLossToOutput[IDX_ZERO].size(); ++i)
        for(auto j=0; j<setGradLossToOutput.size(); ++j)
            for(auto k=0; k<setGradLossToOutput[j][i].ELEM_CNT; ++k)
                vecGradBeta.pos_idx(i) += setGradLossToOutput[j][i].pos_idx(k);
    return vecGradBeta;
}

feature BNDeduce(feature &vecInput, vect &vecBeta, vect &vecGamma, std::shared_ptr<ConvBN> &pBNData, uint64_t iMiniBatchSize = 0, uint64_t iMiniBatchCnt = 0, double dEpsilon = 1e-8)
{
    if(iMiniBatchCnt) for(auto i=0; i<vecInput.size(); ++i)
    {
        pBNData->vecMiuBeta[i] = pBNData->vecMiuBeta[i].elem_cal_opt(iMiniBatchCnt, MATRIX_ELEM_DIV);
        pBNData->vecSigmaSqr[i] = (iMiniBatchSize / (iMiniBatchSize - 1)) * pBNData->vecSigmaSqr[i].elem_cal_opt(iMiniBatchCnt, MATRIX_ELEM_DIV);
    }
    feature vecConvBNDeduceOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i)
    {
        auto vecBarX = (vecInput[i] - pBNData->vecMiuBeta[i]).elem_cal_opt(DIV_DOM(pBNData->vecSigmaSqr[i], dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
        vecConvBNDeduceOutput[i] = (vecGamma.pos_idx(i) * vecBarX).broadcast_add(vecBeta.pos_idx(i));
    }
    return vecConvBNDeduceOutput;
}

vect InitKernelIm2Col(uint64_t iAmt, uint64_t iChannCnt, uint64_t iLnCnt, uint64_t iColCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandAcc = 1e-5) { return vect(iLnCnt*iColCnt*iChannCnt, iAmt, true, dRandBoundryFirst, dRandBoundrySecond, dRandAcc); }

struct ConvBNIm2Col
{
    vect vecIm2ColMuBeta;
    vect vecIm2ColSigmaSqr;
    set<vect> setBarIm2ColX;
    set<vect> setIm2ColY;
    ConvBNIm2Col() {}
    void ValueCopy(ConvBNIm2Col &ConvBNIm2ColVal)
    {
        vecIm2ColMuBeta = ConvBNIm2ColVal.vecIm2ColMuBeta;
        vecIm2ColSigmaSqr = ConvBNIm2ColVal.vecIm2ColSigmaSqr;
        setBarIm2ColX = ConvBNIm2ColVal.setBarIm2ColX;
        setIm2ColY = ConvBNIm2ColVal.setIm2ColY;
    }
    void ValueMove(ConvBNIm2Col &&ConvBNIm2ColVal)
    {
        vecIm2ColMuBeta = std::move(ConvBNIm2ColVal.vecIm2ColMuBeta);
        vecIm2ColSigmaSqr = std::move(ConvBNIm2ColVal.vecIm2ColSigmaSqr);
        setBarIm2ColX = std::move(ConvBNIm2ColVal.setBarIm2ColX);
        setIm2ColY = std::move(ConvBNIm2ColVal.setIm2ColY);
    }
    ConvBNIm2Col(ConvBNIm2Col &ConvBNIm2ColVal) { ValueCopy(ConvBNIm2ColVal); }
    ConvBNIm2Col(ConvBNIm2Col &&ConvBNIm2ColVal) { ValueMove(std::move(ConvBNIm2ColVal)); }
    void operator=(ConvBNIm2Col &ConvBNIm2ColVal) { ValueCopy(ConvBNIm2ColVal); }
    void operator=(ConvBNIm2Col &&ConvBNIm2ColVal) { ValueMove(std::move(ConvBNIm2ColVal)); }
    void Reset()
    {
        vecIm2ColMuBeta.reset();
        vecIm2ColSigmaSqr.reset();
        setBarIm2ColX.reset();
        setIm2ColY.reset();
    }
    ~ConvBNIm2Col() { Reset(); }
};

ConvBNIm2Col BNTrainIm2Col(set<vect> &setIm2ColInput, vect &vecBeta, vect &vecGamma, double dEpsilon = 1e-8)
{
    ConvBNIm2Col BNOutput;
    BNOutput.vecIm2ColMuBeta = (1.0/setIm2ColInput.size()) * setIm2ColInput.sum();
    for(auto i=0; i<setIm2ColInput.size(); ++i)
        if(BNOutput.vecIm2ColSigmaSqr.is_matrix()) BNOutput.vecIm2ColSigmaSqr += (setIm2ColInput[i] - BNOutput.vecIm2ColMuBeta).elem_cal_opt(2, MATRIX_ELEM_POW);
        else BNOutput.vecIm2ColSigmaSqr = (setIm2ColInput[i] - BNOutput.vecIm2ColMuBeta).elem_cal_opt(2, MATRIX_ELEM_POW);
    BNOutput.vecIm2ColSigmaSqr *= (1.0 / setIm2ColInput.size());
    auto vecIm2ColSigma = DIV_DOM(BNOutput.vecIm2ColSigmaSqr, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW);
    BNOutput.setBarIm2ColX.init(setIm2ColInput.size());
    BNOutput.setIm2ColY.init(setIm2ColInput.size());
    for(auto i=0; i<setIm2ColInput.size(); ++i)
    {
        BNOutput.setBarIm2ColX[i] = (setIm2ColInput[i] - BNOutput.vecIm2ColMuBeta).elem_cal_opt(vecIm2ColSigma, MATRIX_ELEM_DIV);
        BNOutput.setIm2ColY[i] = BNOutput.setBarIm2ColX[i];
        for(auto j=0; j<BNOutput.setIm2ColY[i].ELEM_CNT; ++j)
        {
            auto iCurrChann = mtx::mtx_elem_pos(j, BNOutput.setIm2ColY[i].COL_CNT).col;
            BNOutput.setIm2ColY[i].pos_idx(j) *= vecGamma.pos_idx(iCurrChann);
            BNOutput.setIm2ColY[i].pos_idx(j) += vecBeta.pos_idx(iCurrChann);
        }
    }
    return BNOutput;
}

set<vect> BNGradLossToInputIm2Col(set<vect> &setIm2ColGradLossToOutput, ConvBNIm2Col &ConvBNIm2ColOutput, set<vect> &setIm2ColInput, vect &vecGamma, double dEpsilon = 1e-8)
{
    auto vecSigmaSqr = DIV_DOM(ConvBNIm2ColOutput.vecIm2ColSigmaSqr, dEpsilon),
        vecSigma = vecSigmaSqr.elem_cal_opt(0.5, MATRIX_ELEM_POW);
    auto setGradBarX = setIm2ColGradLossToOutput;
    for(auto i=0; i<setIm2ColInput.size(); ++i) for(auto j=0; j<setGradBarX[i].ELEM_CNT; ++j) setGradBarX[i].pos_idx(j) *= vecGamma.pos_idx(mtx::mtx_elem_pos(j, setGradBarX[i].COL_CNT).col);
    vect vecGradSigmaSqr(vecSigma.LN_CNT, vecSigma.COL_CNT);
    for(auto i=0; i<setIm2ColInput.size(); ++i) vecGradSigmaSqr += setGradBarX[i].elem_cal_opt((setIm2ColInput[i]-ConvBNIm2ColOutput.vecIm2ColMuBeta), MATRIX_ELEM_MULT).elem_cal_opt(vecSigmaSqr.elem_cal_opt(1.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
    vecGradSigmaSqr *= (-0.5);
    vect vecDistanceSum(vecSigma.LN_CNT, vecSigma.COL_CNT);
    for(auto i=0; i<setIm2ColInput.size(); ++i) vecDistanceSum += (setIm2ColInput[i] - ConvBNIm2ColOutput.vecIm2ColMuBeta);
    vect vecGradMuBeta = (-1) * setGradBarX.sum().elem_cal_opt(vecSigma, MATRIX_ELEM_DIV) + ((-2.0) / setIm2ColInput.size()) * vecGradSigmaSqr.elem_cal_opt(vecDistanceSum, MATRIX_ELEM_MULT);
    set<vect> setGradInput(setIm2ColInput.size());
    for(auto i=0; i<setGradInput.size(); ++i) setGradInput[i] = setGradBarX[i].elem_cal_opt(vecSigma, MATRIX_ELEM_DIV) + ((2.0) / setIm2ColInput.size()) * vecGradSigmaSqr.elem_cal_opt((setIm2ColInput[i]-ConvBNIm2ColOutput.vecIm2ColMuBeta), MATRIX_ELEM_MULT) + ((1.0) / setIm2ColInput.size()) * vecGradMuBeta;
    return setGradInput;
}

vect BNGradLossToScaleIm2Col(set<vect> &setIm2ColGradLossToOutput, ConvBNIm2Col &ConvBNIm2ColOutput)
{
    vect vecGradGamma(ConvBNIm2ColOutput.vecIm2ColMuBeta.COL_CNT, IDX_SGL);
    for(auto i=0; i<ConvBNIm2ColOutput.setBarIm2ColX.size(); ++i) for(auto j=0; j<ConvBNIm2ColOutput.setBarIm2ColX[i].ELEM_CNT; ++j) vecGradGamma.pos_idx(mtx::mtx_elem_pos(j, ConvBNIm2ColOutput.setBarIm2ColX[i].COL_CNT).col) += setIm2ColGradLossToOutput[i].pos_idx(j) * ConvBNIm2ColOutput.setBarIm2ColX[i].pos_idx(j);
    return vecGradGamma;
}

vect BNGradLossToShiftIm2Col(set<vect> &setIm2ColGradLossToOutput)
{
    vect vecGradBeta(setIm2ColGradLossToOutput[IDX_ZERO].COL_CNT, IDX_SGL);
    for(auto i=0; i<setIm2ColGradLossToOutput.size(); ++i) for(auto j=0; j<setIm2ColGradLossToOutput[i].ELEM_CNT; ++j) vecGradBeta.pos_idx(mtx::mtx_elem_pos(j, setIm2ColGradLossToOutput[i].COL_CNT).col) += setIm2ColGradLossToOutput[i].pos_idx(j);
    return vecGradBeta;
}

void BNDeduceIm2ColInit(BN_EXP_VAR &BNData, uint64_t iBatchCnt, uint64_t iBatchSize)
{
    if(iBatchCnt)
    {
        BNData.vecExp = BNData.vecExp.elem_cal_opt(iBatchCnt, MATRIX_ELEM_DIV);
        BNData.vecVar = BNData.vecVar.elem_cal_opt(iBatchCnt, MATRIX_ELEM_DIV);
        if(iBatchSize > 1) BNData.vecVar *= (iBatchSize / (iBatchSize - 1.0));
    }
}

vect BNDeduceIm2Col(vect &vecIm2ColInput, vect &vecBeta, vect &vecGamma, BN_EXP_VAR &BNData, double dEpsilon = 1e-8)
{
    auto vecNrom = vecIm2ColInput - BNData.vecExp,
        vecVar = DIV_DOM(BNData.vecVar, dEpsilon);
    vect vecAns = vecNrom.elem_cal_opt(vecVar.elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
    for(auto i=0; i<vecAns.ELEM_CNT; ++i)
    {
        auto iCurrChann = mtx::mtx_elem_pos(i, vecAns.COL_CNT).col;
        vecAns.pos_idx(i) = vecGamma.pos_idx(iCurrChann) * vecAns.pos_idx(i) + vecBeta.pos_idx(iCurrChann);
    }
    return vecAns;
}

CONV_END