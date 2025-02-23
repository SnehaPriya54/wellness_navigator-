const express = require("express");

const app = express();
const PORT = 5000;
//app.listen(port,()=> console.log("Listening on port $(port)"))

app.set('view engine','ejs');

app.listen(PORT, function(err){
    if (err) console.log("Error in server setup")
    console.log("Server listening on Port", PORT);
})

//index
app.get("/",(req,res)=>{
    res.render("index",{title:"HOME PAGE"})
})

app.get("/search",(req,res)=>{
    res.render("search",{title:"SEARCH"})
})

app.get("/about",(req,res)=>{
    res.render("about",{title:"ABOUT US"})
})

app.use(express.static('image'));
app.use(express.urlencoded({ extended: true}) )

