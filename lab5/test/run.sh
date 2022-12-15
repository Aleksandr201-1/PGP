#echo $1
./testgen test3.txt ans3.txt $1
./../lab5 < test3.txt > a3.txt
diff a3.txt ans3.txt