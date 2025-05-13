const express=require('express');
const axios=require('axios')
const userModel=require('../models/userModel')
const patientModel=require('../models/patientModel')
const patientRecordModel=require('../models/patientRecordModel')
const {validateToken} = require('../middleware/handleTokens')
const mongoose = require('mongoose');

const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
// Set up Multer to handle file uploads
const storage = multer.memoryStorage();  // We are using memory storage to keep the file in memory
const upload = multer({ storage: storage });

async function processWithMLModel(recordId, files) {
  try {
    const form = new FormData();
    form.append('id', 123);
    files.forEach((file) => {
      form.append('files', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
      });
    });

    const flaskResponse = await axios.post('https://hacky454-ecgAnalyser.hf.space/predict',
      form,
      { headers: { ...form.getHeaders() }, timeout: 1000 * 600 } // 10 min timeout
    );

    const data = flaskResponse.data;
    console.log('ML model response:', data);

    await patientRecordModel.updateOne({ _id:recordId }, { result: data,isGenerated:true });

    console.log(`Result saved for patient ${recordId}`);
  } catch (error) {
    console.error('ML processing failed:', error);
  }
}

function validateUploadedFiles(files) {
  if (!files || files.length > 4) {
    return 'A maximum of 4 files is allowed';
  }

  const baseNames = new Set();
  const extensions = new Set();

  for (let file of files) {
    const parts = file.originalname.split('.');
    if (parts.length < 2) {
      return 'Invalid file name format. Must be name.ext';
    }
    const ext = parts.pop().toLowerCase();
    const base = parts.join('.');
    baseNames.add(base);
    extensions.add(ext);
  }

  if (!(extensions.has('dat') && extensions.has('hea'))) {
    return 'Both .dat and .hea files are required';
  }

  if (baseNames.size !== 1) {
    return 'All files must have the same base name';
  }

  return null; // Valid
}


const ecgRouter=express.Router();

//this is to list all patients of a particular doctor
ecgRouter.get('/patients', validateToken, async (req, res, next) => {
  try {
    const doctor = await userModel.findOne({ email: req.user.email });
    const { search = '', sort = 'patientId',order = 'asc' } = req.query;
    // Base filter: only patients for this doctor
    const filter = { doctorId: doctor._id };

    // If search is provided, filter by name (case-insensitive)
    if (search) {
      filter.name = { $regex: search, $options: 'i' };
    }

    // Define sort condition
    let sortCondition = {};
    const sortOrder= order=='asc'?1:-1
    sortCondition = {[sort]:sortOrder}

    const patients = await patientModel.find(filter, {age: 1,gender: 1,name: 1,_id: 1,patientId:1 }).sort(sortCondition);

    return res.status(200).json({ msg: 'Patients Info', patients });
  } catch (err) {
    next(err);
  }
});

ecgRouter.get('/patients/:id', validateToken, async (req, res, next) => {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) return res.status(400).json({ message: 'Invalid patient ID' });

    const doctor = await userModel.findOne({ email: req.user.email });

    const patient = await patientModel.findOne({ _id: id, doctorId: doctor._id }, {
      age: 1, gender: 1, name: 1, _id: 1,patientId: 1,
    });

    if (!patient) return res.status(404).json({ message: 'Patient not found or unauthorized access.' });

    const records = await patientRecordModel.find({ patientId: id }, {
      _id: 1, isGenerated: 1, createdAt: 1,
    });

    return res.status(200).json({ message: 'Records fetched successfully.', records, patient });
  } catch (err) {
    next(err);
  }
});

ecgRouter.get('/records/:recordId', validateToken, async (req, res, next) => {
  try {
    const { recordId } = req.params;
    if (!mongoose.Types.ObjectId.isValid(recordId)) return res.status(400).json({ message: 'Invalid record ID' });

    const doctor = await userModel.findOne({ email: req.user.email });

    const record = await patientRecordModel.findOne({ _id: recordId, doctorId: doctor._id });

    if (!record) return res.status(404).json({ message: 'Record not found or unauthorized access.' });

    return res.status(200).json({ message: 'Record fetched successfully.', record });
  } catch (err) {
    next(err);
  }
});

ecgRouter.delete('/patients/:id', validateToken, async (req, res, next) => {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) return res.status(400).json({ msg: 'Invalid patient ID' });

    const doctor = await userModel.findOne({ email: req.user.email });

    const patient = await patientModel.findOne({ _id: id, doctorId: doctor._id });
    if (!patient) return res.status(404).json({ msg: 'Patient not found or unauthorized access.' });

    await patientRecordModel.deleteMany({ patientId: id });
    await patientModel.deleteOne({ _id: id });

    return res.status(200).json({ msg: 'Patient and their ECG records deleted successfully' });
  } catch (err) {
    next(err);
  }
});

ecgRouter.delete("/records/:recordId", validateToken, async (req, res, next) => {
  try {
    const { recordId } = req.params;
    if (!mongoose.Types.ObjectId.isValid(recordId)) return res.status(400).json({ msg: 'Invalid record ID' });

    const doctor = await userModel.findOne({ email: req.user.email });

    const record = await patientRecordModel.findOne({ _id: recordId, doctorId: doctor._id });
    if (!record) return res.status(404).json({ msg: 'Record not found or unauthorized access.' });

    await patientRecordModel.deleteOne({ _id: recordId });

    return res.status(200).json({ msg: 'ECG record deleted successfully' });
  } catch (err) {
    next(err);
  }
});

ecgRouter.post('/upload',validateToken, upload.array('files'), async (req, res) => {
    console.log('inside uplod');
    const { name, age, gender } = req.body;
    let { patientId } = req.body
    const files = req.files; 
    const errLog=validateUploadedFiles(files)
    if(errLog)  return res.status(404).json({ msg: errLog });
    const doctor= await userModel.findOne({email:req.user.email});
    console.log(doctor);
    console.log('Form Data:', { name, age, gender });
    // console.log('Files:', files);
    try {
      if (!files) {
        return res.status(400).send({ message: 'No file uploaded' });
      }

      if(patientId){
        if (!mongoose.Types.ObjectId.isValid(patientId)) {
          return res.status(400).json({ msg: 'Invalid patient ID format' });
        }

        const patient = await patientModel.findById(patientId);
        if (!patient) {
          return res.status(404).json({ msg: 'Patient not found with provided ID' });
        }
        console.log('patient id  is '+patientId);
      }else{
        console.log('new patient');
        let newCount=doctor.patientCount+1
        console.log(newCount);
        await userModel.updateOne({_id:doctor._id},{patientCount:newCount})
        const patient = await patientModel.create({name,age,gender,doctorId:doctor._id})
        console.log(patient);
        patientId=patient._id;
      }

      const newRecord= await patientRecordModel.create({patientId,doctorId:doctor._id})
      console.log(newRecord);
      processWithMLModel(newRecord._id,files);
      res.status(200).json({ msg:'Ecg Files uploaded successfully' });

    } catch (error) {
      console.error('Error uploading file:', error);
      res.status(500).send({ message: 'Internal Server Error' });
    }
});

module.exports=ecgRouter;


// i think not just 1 several get routes req
//1. to return the result of uploaded file
//2. to give list of patient in dashboard   -> patientName, age, gender
//3. to give list of patient record of a particular user -> not  full -> recordNo, date

// see of each upload u have to create patietnRecord
// onlly one thing is isseu that is patientModel, hwo we are going to handle existing vs new 
// if you are able to do line 25 then evverything will fall on your way


    // const accessToken=jwt.sign({email:email},process.env.ACCESS_TOKEN, {expiresIn:'3m'});
    // const refreshToken=jwt.sign({email:email},process.env.REFRESH_TOKEN, {expiresIn:'15m'});
    // res.cookie('accessToken',accessToken,{
    //     httpOnly:true,secure:true,sameSite:'strict',maxAge:60000
    // });
    // res.cookie('refreshToken',refreshToken, {
    //     httpOnly:true,secure:true,sameSite:'strict',maxAge:300000
    // });
    // console.log('cookies is ');


// adminRoutes.post('/admin-login',async(req,res,next)=>{
//     const {email,passwd}=req.body;
//     console.log(req.body);
//     if(email==='' || passwd==='')return res.send({status:400,msg:"Input fields are missing"})
//     const admindata= await adminModel.findOne({email:email});
//     console.log(admindata);
//     if(admindata==null) return res.send({status:404,msg:'So such admin exist'});
//     if(!comparePassword(passwd,admindata.passwd)) return res.send({status:300,msg:'Invalid credentials'});
//     //create token

//     const accessToken=jwt.sign({email:email},process.env.ACCESS_TOKEN, {expiresIn:'3m'});
//     const refreshToken=jwt.sign({email:email},process.env.REFRESH_TOKEN, {expiresIn:'15m'});
//     res.cookie('accessToken',accessToken,{
//         httpOnly:true,secure:true,sameSite:'strict',maxAge:60000
//     });
//     res.cookie('refreshToken',refreshToken, {
//         httpOnly:true,secure:true,sameSite:'strict',maxAge:300000
//     });
//     console.log('cookies is ');
//     return res.send({status:200,msg:'success'})
// })

// adminRoutes.post('/admin-logout',verifyToken,(req,res,next)=>{
//     delete req.email;
//     res.clearCookie('accessToken');
//     res.clearCookie('refreshToken')
//     return res.send({status:200,msg:'Logout success'});
// })

// adminRoutes.post('/admin-edit',verifyToken,async(req,res,next)=>{
//     console.log('in /admin-edit');
//     console.log(req.body);
//     const {fname,lname,email,passwd}=req.body;
//     if(fname===''||lname===''||email===''||passwd==='') return res.send({status:400,msg:'Some input fields are missing'})
//     const passwd2=hashPassword(passwd)
//     console.log(passwd2);
//     try{
//         await adminModel.findOneAndUpdate({email:email},{$set:{
//             fname,lname,email,passwd:passwd2
//         }});
//         const adminDetails=await adminModel.findOne({email:email})
//         return res.send({status:200,msg:'Profile info updated successfully!',data:adminDetails})         
//     }catch(err){
//         console.log(err);
//         return res.send({status:300,msg:err})
//     }
// })

// adminRoutes.get('/get-admindata',verifyToken,async(req,res,next)=>{
//     console.log('entered get-admindata');
//     const email=req.email
//     delete req.email;
//     console.log(email);
//     const adminData=await adminModel.findOne({email:email});
//     //if adminData is empty
//     if(adminData===null) return res.send({status:404,msg:'No such user in database'})
//     console.log('admin data is ');
//     console.log(adminData);
//     return res.send({status:200,msg:'authorized',data:adminData})
// })


// const adminModel=require('../models/adminModel');
// const verifyToken=require('../auth/adminAuth')

// authRoute.get('/hi',async(req,res,next)=>{
//     console.log('entered hi route');
//     res.send('Hi dude')
    
// })


    // axios.post('http://127.0.0.1:5000/demo',JSON.stringify(newUser),
    // {headers: {
    //     'Content-Type': 'application/json',  // Specify content type as JSON
    //   },}
    // )
    // .then(res=>res.data)
    // .then(data=>{
    //     console.log(data);
    // })
    // .catch(err=>{
    //     console.log(err);
    //     console.log('no response from flask server')
    // })
    
    // console.log(newUser);
    // console.log(user._id?true:false);
   /*
   ecgRouter.post('/createAccount',async(req,res,next)=>{
    console.log(req.body)
    const {name,email,pass,phone}=req.body
    const password= hashPassword(pass)
    console.log(password);

    const user=await userModel.findOne({email:email})
    console.log((user?'user is present':'user is not present '));
    
    if(user) return res.status(300).send({msg:'Account with same email already exist'})

    const newUser=await userModel.create({name,email,phone,password})
    console.log(newUser)
    
    await generateToken(res,email)
    return res.send({status:200,msg:'success'})
 
})

ecgRouter.post('/login',async(req,res,next)=>{
    console.log('inside login');
    console.log(req.body);
    const {email,pass}=req.body
    const user= await userModel.findOne({email:email})
    if(!user) return res.status(300).send({msg:'No such user exist'})
    const isValidPass=comparePassword(pass,user.password)
    if(! isValidPass) return res.status(400).send({msg:'Invalid credentials'})
    console.log(user);

    await generateToken(res,email)

    res.status(200).send({name:user.name,email:user.email,phone:user.phone})
})

ecgRouter.post('/logout',async(req,res,next)=>{
    res.clearCookie("accessToken", {
        httpOnly: true,
        secure: true,
        sameSite: "strict",
    });
    res.clearCookie("refreshToken", {
        httpOnly: true,
        secure: true,
        sameSite: "strict",
    });
    
    return res.status(200).json({ msg: "Logged out successfully" });
})

   */

/*

ecgRouter.delete("/records/:recordId",validateToken,async (req,res,next)=>{
  const {recordId} = req.params;
  const result = await patientRecordModel.deleteOne({_id:recordId});
  if (result.deletedCount === 0) {
    return res.status(404).json({ msg: 'No patient found with that ID' });
  }
  return res.status(200).json({msg:'Ecg report deleted successfully'})
})
*/


/*

ecgRouter.delete('/patients/:id',validateToken, async (req,res,next)=>{
  const { id } = req.params;
  await patientRecordModel.deleteMany({patientId: id})
  const result = await patientModel.deleteOne({ _id: id });
  console.log(result);
  if (result.deletedCount === 0) {
    return res.status(404).json({ msg: 'No patient found with that ID' });
  }
  return res.status(200).json({msg:'Patient Info and their ecg reports deleted successfully'})
})

*/

      //https://ecg-analyser.onrender.com/predict
      //.hf.space/predict
      // https://hacky454-ecganalyser.hf.space/predict
      // https://huggingface.co/spaces/hacky454/ecgAnalyser/predict


      // fs.writeFile('response.json', JSON.stringify(data, null, 2), (err) => {
      //   if (err) {
      //     console.error('Error writing file:', err);
      //   } else {
      //     console.log('Saved response to response.json');
      //   }
      // });

// const flaskResponse = await axios.post('https://hacky454-ecgAnalyser.hf.space/predict', form, {
      //   headers: {
      //     ...form.getHeaders(),  
      //   }, timeout: 1000 * 300
      // });
      // console.log(flaskResponse.data);
      // const data = flaskResponse.data;
      // console.log('Response:', data);
      //insert into mongodb

      // write JSON to a file
      
    //   return  res.status(200).json({ flaskResponse: flaskResponse.data });
          

// ecgRouter.get('/patients',validateToken,async(req,res,next)=>{
//   const doctor= await userModel.findOne({email:req.user.email});
//   // const doctorId=doctor._id
//   const patients= await patientModel.find({doctorId:doctor._id},{age:1,gender:1,name:1,_id:1})
//   console.log(patients);
//   return  res.status(200).json({ msg:'Patients Info',patients });
// })


// async function processWithMLModel(){
//   const form = new FormData();
//   form.append('id',123)
//   files.forEach((file) => {
//     form.append('files', file.buffer, {
//       filename: file.originalname,
//       contentType: file.mimetype,
//     });
//   });

// }


//this is to list all records of a particular patient, partial detail is enough
// ecgRouter.get('/patients/:id',validateToken,async(req,res,next)=>{
//   // const doctor= await userModel.findOne({email:req.user.email});
//   console.log(req.params)
//   const {_id}=req.user
//   const {id}=req.params;
//   const records= await patientRecordModel.find({patientId:id})
//   console.log(records   );
// })

//this is the page where specific record is given with full detail
// ecgRouter.get('/records/:recordId',async(req,res,next)=>{
//   console.log(req.params);
//   const {recordId}= req.params;
//   const record=await patientRecordModel.findOne({_id:recordId})
//   console.log(record);
// })

// few get request needed to search, also get 

// i think its not needed, as simply a get requet for each record, 
// so based on presence / absence of field in front end itself we could handle no need of this reeq
// //here we assume that user sends a list of ecgrecord's _id field in order to check are they finished if so we fetch them and send as response
// // which will update the result in front pae
// ecgRouter.post('/getPredictions',async(req,res,next)=>{

// })

/*


ecgRouter.get('/records/:recordId', validateToken, async (req, res, next) => {
  try {
    const { recordId } = req.params;

    const record = await patientRecordModel.findById(recordId);

    if (!record)  return res.status(404).json({ message: 'Record not found.' });
    return res.status(200).json({ message: 'Record fetched successfully.', record });
  } catch (err) {
    console.error('Error fetching record:', err);
    next(err);
  }
});
*/

/*

ecgRouter.get('/patients/:id', validateToken, async (req, res, next) => {
  try {
    const { id } = req.params;
    const patient = await patientModel.findOne({_id:id}, {age: 1,gender: 1,name: 1,_id: 1,patientId:1 });
    const records = await patientRecordModel.find({ patientId: id }, {
      _id: 1,
      isGenerated:1,
      createdAt: 1, 
    });

    if (records.length === 0) {
      return res.status(404).json({ message: 'No records found for this patient.' });
    }

    return res.status(200).json({ message: 'Records fetched successfully.', records,patient });
  } catch (err) {
    console.error('Error fetching patient records:', err);
    next(err);
  }
});

*/