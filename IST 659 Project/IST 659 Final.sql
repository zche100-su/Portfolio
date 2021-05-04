CREATE TABLE Employee_C ( 
	Employee_ID VARCHAR(10) PRIMARY KEY, 
	Supermarket_ID VARCHAR(10) NOT NULL,
	Emp_SSN VARCHAR(11), 
	Emp_FName VARCHAR(25), 
	Emp_LName VARCHAR(25) NOT NULL, 
	Emp_Position VARCHAR(10) NOT NULL check(Emp_Position IN ('Internship','Manager','Employee')), 
	Emp_Address1 VARCHAR(30)NOT NULL, 
	Emp_Address2 VARCHAR(30), 
	Emp_Phone VARCHAR(12) NOT NULL, 
	Emp_Email VARCHAR(50) NOT NULL, 
	Emp_Start_Date DATE NOT NULL, 
	Emp_End_Date DATE, 
) 

insert into Employee_C Values('E000001','S0001','123-4566-45','Jay','Macie','Manager','Big Palace','','315-372-0047','JM123@Super.com','04-13-2019','');
select * from Employee_C

CREATE TABLE Manager_C( 
	Manager_ID VARCHAR(10) PRIMARY KEY, 
	Employee_ID VARCHAR(10) NOT NULL,
	Supermarket_ID VARCHAR(10) NOT NULL, 
	M_Description VARCHAR(99) NOT NULL, 
 ) 
insert into Manager_C Values('M0001','E000001','S0001','a good manager');
select * from Manager_C

CREATE TABLE Supermarket_C( 
	Supermarket_ID VARCHAR(10) PRIMARY KEY, 
	Manager_ID VARCHAR(10) NOT NULL,
	S_City VARCHAR(30) NOT NULL, 
	S_State VARCHAR(2) NOT NULL, 
	S_Address1 VARCHAR(30) NOT NULL, 
	S_Address2 VARCHAR(30), 
) 
insert into Supermarket_C Values('S0001','M0001','Syracuse','NY','Comstock Ave','123');
select * from Supermarket_C

CREATE TABLE Product_Type_C( 
	Product_Type_ID VARCHAR(10) PRIMARY KEY, 
	Product_Name VARCHAR(10)  NOT NULL,
	Product_Type_Price VARCHAR(10) NOT NULL, 
	Amount_in_Stock INTEGER NOT NULL,
)
insert into Product_Type_C Values('PT000001','Big Apple','5','5000');
select * from Product_Type_C

CREATE TABLE Product_C( 
	Product_ID VARCHAR(10) PRIMARY KEY, 
	Product_Type_ID VARCHAR(10) NOT NULL,
	Product_Type_Price INTEGER NOT NULL, 
	Supermarket_ID VARCHAR(10) NOT NULL, 
	Product_Name VARCHAR(30) NOT NULL, 
	Date_of_Manufacture DATE NOT NULL, 
	Expiration_Date DATE NOT NULL, 
	Product_Price INTEGER NOT NULL, 
)
insert into Product_C Values('P000000001','PT000001','5','S0001','Big Apple','10-13-2019','12-13-2019','4');
select * from Product_C

CREATE TABLE Warehouse_C( 
	Warehouse_ID VARCHAR(10) PRIMARY KEY, 
	WH_Phone VARCHAR(12) NOT NULL,
	WH_Email VARCHAR(50) NOT NULL, 
	WH_City VARCHAR(30) NOT NULL, 
	WH_State VARCHAR(2) NOT NULL, 
	WH_Address_1 VARCHAR(30) NOT NULL, 
	WH_Address_2 VARCHAR(30), 
) 
insert into Warehouse_C Values('WH00001 ','315-372-0001','WH00001@S.com','Syracuse','NY','101 A Nice Street','');
select * from Warehouse_C

CREATE TABLE Replenish_Order_C( 
	Rep_Order_ID VARCHAR(10) PRIMARY KEY, 
	Manager_ID VARCHAR(10) NOT NULL,
	Warehouse_ID VARCHAR(10) NOT NULL, 
	Product_Type_ID VARCHAR(10) NOT NULL,
	Amount INTEGER NOT NULL,
)
insert into Replenish_Order_C Values('R000000001','M0001','WH00001','PT000001','5000');
select * from Replenish_Order_C
/*
drop table Product_Type_Z
drop table Employee_Z
drop table Manager_Z
drop table Product_Z
drop table Supermarket_Z
drop table Replenish_Order_Z
drop table Warehouse_Z
*/




ALTER TABLE Employee_C
ADD FOREIGN KEY (Supermarket_ID)
REFERENCES Supermarket_C(Supermarket_ID)


ALTER TABLE Supermarket_C
ADD FOREIGN KEY (Manager_ID)
REFERENCES Manager_C(Manager_ID)


ALTER TABLE Manager_C
ADD FOREIGN KEY (Employee_ID)
REFERENCES Employee_C(Employee_ID)

ALTER TABLE Manager_C
ADD FOREIGN KEY (Supermarket_ID)
REFERENCES Supermarket_C(Supermarket_ID)


ALTER TABLE Product_C
ADD FOREIGN KEY (Product_Type_ID)
REFERENCES Product_Type_C(Product_Type_ID)


ALTER TABLE Product_C
ADD FOREIGN KEY (Supermarket_ID)
REFERENCES Supermarket_C(Supermarket_ID)


ALTER TABLE Replenish_Order_C
ADD FOREIGN KEY (Manager_ID)
REFERENCES Manager_C(Manager_ID)


ALTER TABLE Replenish_Order_C
ADD FOREIGN KEY (Warehouse_ID)
REFERENCES Warehouse_C(Warehouse_ID)


ALTER TABLE Replenish_Order_C
ADD FOREIGN KEY (Product_Type_ID)
REFERENCES Product_Type_C(Product_Type_ID)





