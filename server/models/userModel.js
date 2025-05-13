const mongoose = require('mongoose')

const userSchema=new mongoose.Schema({
    'name':{
        type:String,
        required:true
    },
    'email':{
        unique:true,
        type:String,
        required:true
    },
    'password':{
        type:String,
        required:true
    },
    'phone':{
        type:String,
        required:true
    },
    'patientCount':{
        type:Number,
        default:0
    }
})

const userModel=mongoose.model('user',userSchema,'user')

module.exports=userModel;