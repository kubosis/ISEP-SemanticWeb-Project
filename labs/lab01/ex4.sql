CREATE TABLE Courses(
 code CHAR(5) PRIMARY KEY,
 credits INTEGER,
 title VARCHAR(64) UNIQUE,
 description VARCHAR(1024),
 professor VARCHAR(64),
)