const mongoose = require('mongoose')

const patientRecordSchema = new mongoose.Schema({
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'patient', required: true },
    doctorId: { type: mongoose.Schema.Types.ObjectId, ref: 'user', required: true },
  
    createdAt: { type: Date, default: Date.now },
    isGenerated:{
        type:Boolean,
        default:false
    },
    result:{
        type:Object,
        default:{}
    }
});
  
const patientRecordModel=mongoose.model('patientRecord',patientRecordSchema,'patientRecord')
module.exports=patientRecordModel;
  
/*
    // AI-generated results
    diagnosisResult: String,
    riskScore: Number,
    otherMetrics: Object, // or use specific fields if you know the
*/
