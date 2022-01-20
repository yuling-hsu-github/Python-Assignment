
import csv
fn1='DOITT_MINY_VENDOR_01_13SEPT2010.csv'
with open(fn1, encoding='utf8')  as file:
    csvReader=csv.reader(file)     
    vendor=list(csvReader)
    
import sqlite3
conn= sqlite3.connect("vendor1.sqlite")
sqlstr='''CREATE TABLE if not exists vendor1("ID" INTEGER NOT NULL, "VENDORNAME" TEXT,"CITY" TEXT, "CATEGORY" TEXT ); '''
cursor=conn.execute(sqlstr)

for d in range(50,len(vendor)):    
    sql="insert into vendor1(ID,VENDORNAME,CITY,CATEGORY) values('{0}','{1}','{2}','{3}')"
    sql1=sql.format(vendor[d][1],vendor[d][3],vendor[d][7],vendor[d][2])
    print(sql1)
    cursor=conn.execute(sql1)
    conn.commit()
conn.close()
#要用lens
#VENDORNAME是因為兩層list