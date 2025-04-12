FC_BEGIN

vect Output(vect &vecInput, vect &vecWeight) {return vecWeight * vecInput;}

vect GradLossToInput(vect &vecGradLossToOutput, vect &vecWeight) {return vecWeight.transposition() * vecGradLossToOutput;}

vect GradLossToWeight(vect &vecGradLossToOutput, vect &vecInput) {return vecGradLossToOutput * vecInput.transposition();}

vect FeatureTransform(feature &vecInput)
{
    auto iElemCnt = vecInput[IDX_ZERO].ELEM_CNT,
        iChannCnt = vecInput.size(),
        iLnCnt = iChannCnt * iElemCnt,
        iCpyCnt = 0Ui64;
    auto vecResTransForm = vect(iLnCnt, 1);
    for(auto i=0; i<iChannCnt; ++i) for(auto j=0; j<iElemCnt; ++j) vecResTransForm.pos_idx(iCpyCnt++) = vecInput[i].pos_idx(j);
    return vecResTransForm;
}

feature FeatureTransform(vect &vecInput, uint64_t iLnCnt, uint64_t iColCnt)
{
    auto iElemCnt = iLnCnt * iColCnt,
        iChannCnt = vecInput.LN_CNT / iElemCnt,
        iCpyCnt = 0Ui64;
    feature vecResTransForm(iChannCnt);
    for(auto i=0; i<iChannCnt; ++i)
    {
        vecResTransForm[i] = vect(iLnCnt, iColCnt);
        for(auto j=0; j<iElemCnt; ++j) vecResTransForm[i].pos_idx(j) = vecInput.pos_idx(iCpyCnt++);
    }
    return vecResTransForm;
}

vect FeatureTransformIm2Col(vect &vecIm2ColInput) { return vecIm2ColInput.reshape(vecIm2ColInput.ELEM_CNT, IDX_SGL); }

vect FeatureTransformIm2Col(vect &vecIm2ColInput, uint64_t iLnCnt, uint64_t iColCnt)
{
    auto iElemCnt = iLnCnt * iColCnt;
    return vecIm2ColInput.reshape(iElemCnt, vecIm2ColInput.LN_CNT/iElemCnt);
}

FC_END