N=10000
FNAME=data.jsonl

#echo "Generating list_maximum data for ints of size $N"
#python gen_data.py --task list_maximum --min 0 --max 99 -n $N -m 5 --type ints --fname $FNAME
#python gen_data.py --task list_maximum --min 0 --max 999 -n $N -m 5 --type ints --fname $FNAME
#python gen_data.py --task list_maximum --min 0 --max 9999 -n $N -m 5 --type ints --fname $FNAME
#python gen_data.py --task list_maximum --min 0 --max 9999 -n $N -m 5 --type ints --fname $FNAME
#
#echo "Generating list_maximum data for floats of size $N"
#python gen_data.py --task list_maximum --min 0 --max 99 -n $N -m 5 --type floats --fname $FNAME
#python gen_data.py --task list_maximum --min 0 --max 999 -n $N -m 5 --type floats --fname $FNAME
#python gen_data.py --task list_maximum --min 0 --max 9999 -n $N -m 5 --type floats --fname $FNAME
#python gen_data.py --task list_maximum --min 0 --max 9999 -n $N -m 5 --type floats --fname $FNAME
#
#echo "Generating list_maximum data for words of size $N"
#python gen_data.py --task list_maximum --min 0 --max 99 -n $N -m 5 --type words --fname $FNAME
#
#echo "Generating decoding data for ints of size $N"
#python gen_data.py --task decoding --min 0 --max 99 -n $N -m 1 --type ints --fname $FNAME
#python gen_data.py --task decoding --min 0 --max 999 -n $N -m 1 --type ints --fname $FNAME
#python gen_data.py --task decoding --min 0 --max 9999 -n $N -m 1 --type ints --fname $FNAME
#python gen_data.py --task decoding --min 0 --max 9999 -n $N -m 1 --type ints --fname $FNAME
#
#echo "Generating decoding data for floats of size $N"
#python gen_data.py --task decoding --min 0 --max 99 -n $N -m 1 --type floats --fname $FNAME
#python gen_data.py --task decoding --min 0 --max 999 -n $N -m 1 --type floats --fname $FNAME
#python gen_data.py --task decoding --min 0 --max 9999 -n $N -m 1 --type floats --fname $FNAME
#python gen_data.py --task decoding --min 0 --max 9999 -n $N -m 1 --type floats --fname $FNAME
#
#echo "Generating decoding data for words of size $N"
#python gen_data.py --task decoding --min 0 --max 99 -n $N -m 1 --type words --fname $FNAME
#
#echo "Generating addition data for ints of size $N"
#python gen_data.py --task addition --min 0 --max 99 -n $N -m 5 --type ints --fname $FNAME
#python gen_data.py --task addition --min 0 --max 999 -n $N -m 5 --type ints --fname $FNAME
#python gen_data.py --task addition --min 0 --max 9999 -n $N -m 5 --type ints --fname $FNAME
#python gen_data.py --task addition --min 0 --max 9999 -n $N -m 5 --type ints --fname $FNAME
#
#echo "Generating addition data for floats of size $N"
#python gen_data.py --task addition --min 0 --max 99 -n $N -m 5 --type floats --fname $FNAME
#python gen_data.py --task addition --min 0 --max 999 -n $N -m 5 --type floats --fname $FNAME
#python gen_data.py --task addition --min 0 --max 9999 -n $N -m 5 --type floats --fname $FNAME
#python gen_data.py --task addition --min 0 --max 9999 -n $N -m 5 --type floats --fname $FNAME
#
#echo "Generating addition data for words of size $N"
#python gen_data.py --task addition --min 0 --max 99 -n $N -m 5 --type words --fname $FNAME

echo "Generating logic data for bool_simple of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type bool_simple --fname $FNAME
echo "Generating logic data for bool of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type bool --fname $FNAME
echo "Generating logic data for algebraic_int of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type algebraic_int --fname $FNAME
echo "Generating logic data for algebraic_float of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type algebraic_float --fname $FNAME
echo "Generating logic data for var+algebraic_int of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type var+algebraic_int --fname $FNAME
echo "Generating logic data for var+algebraic_float of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type var+algebraic_float --fname $FNAME
echo "Generating logic data for var+comparative_int of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type var+comparative_int --fname $FNAME
echo "Generating logic data for var+comparative_float of size $N"
python gen_data.py --task logic --min 0 --max 99 -n $N -m 6 --type var+comparative_float --fname $FNAME