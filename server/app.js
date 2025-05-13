require('dotenv').config();

const express=require('express');
const axios=require('axios')
const cors=require('cors')
const mongoose=require('mongoose');
const cookieParser=require('cookie-parser');
const bodyParser=require('body-parser')
// Sg4hAWLJ0RYwKCg9
//const adminModel=require('./models/adminModel')
// mongoose.connect('mongodb://127.0.0.1:27017/finalYearProject')
mongoose.connect(process.env.MONGO_URL)
  .then(() => console.log("MongoDB connected"))
  .catch(err => console.error("Connection error:", err));


const app=express();
app.use(cors({
    methods:['POST','GET','DELETE'],
    credentials:true,
    origin:[
      'http://localhost:4201', 'http://127.0.0.1:4201',
      'http://localhost:4200', 'http://127.0.0.1:4200',  
      'http://localhost:5000', 'http://127.0.0.1:5000',
      'http://localhost:5173', 'http://127.0.0.1:5173',
      'https://ecganalyser.netlify.app/'
    ]
}))
app.use(cookieParser())
app.use(bodyParser.urlencoded({extended:true}))
app.use(express.json());

app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ message: 'Something went wrong', error: err.message });
});

app.use('/',require('./routes/authRoute'))
app.use('/',require('./routes/ecgRoutes'))
  

app.listen(3000,()=>{
    console.log('the app is running on port 3000');
})