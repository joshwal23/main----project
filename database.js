const mysql = require("mysql2");

const db = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "yourpassword",
    database: "fitquest"
});

module.exports = db;
