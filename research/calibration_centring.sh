#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction
sed -i 's/^from scipy.ndimage import gaussian_filter1d$/from scipy.ndimage import gaussian_filter1d\nfrom sklearn.isotonic import IsotonicRegression/' margin_distribution.py 2>/dev/null
grep -q 'IsotonicRegression' margin_distribution.py && echo "import present"
python -m py_compile margin_distribution.py && echo "COMPILE OK"
echo
echo "############ RAW CENTRE ############"
python margin_distribution.py --raw-centre --validate
echo
echo "############ CALIBRATED CENTRE ############"
python margin_distribution.py --validate
