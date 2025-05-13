const mongoose = require('mongoose')

const patientSchema = new mongoose.Schema({
    doctorId: { type: mongoose.Schema.Types.ObjectId, ref: 'user', required: true },
  
    patientId: {
      type: Number,
      default:0
    },
  
    name: String,
    age: Number,
    gender: { type: String, enum: ['Male', 'Female', 'Other'] },
  });
  
const patientModel=mongoose.model('patient',patientSchema,'patient')
module.exports=patientModel;
// const mongoose = require('mongoose')

// const patientSchema=new mongoose.Schema({
//     'fname':{
//         type:String,
//         required:true
//     },
//     'lname':{
//         type:String,
//         required:true
//     },
//     'email':{
//         type:String,
//         required:true
//     },
//     'password':{
//         type:String,
//         required:true
//     }
// })

// const patientModel=mongoose.model('user',patientSchema,'user')

// module.exports=patientModel;