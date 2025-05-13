require('dotenv').config()
const jwt=require('jsonwebtoken');

//it is a function
function renewToken(req,res,next){
    const {refreshToken}=req.cookies;
    let exist=false;
    let response={};
    if(!refreshToken){
        response={status:405,valid:false,msg:'Refresh token expired'}
    }else{
        jwt.verify(refreshToken,process.env.REFRESH_TOKEN,async(err,decoded)=>{
            if(err){
                console.log(err);
                console.log('invalid refresh token');
                response={msg:'invalid refresh token',status:404,valid:false}
            }else{
                console.log('valid refresh token');         
                console.log(decoded.email);
                const accessToken=  jwt.sign({email:decoded.email},
                    process.env.ACCESS_TOKEN, {expiresIn:'1m'}
                );
                console.log('accessToken is '+accessToken);
                req.email=decoded.email
                console.log('decoded.email from renewToken() '+decoded.email);
                exist=true;     
                response={msg:'success'}               
                res.cookie('accessToken',accessToken,{
                    maxAge:'60000',httpOnly:'true',secure:true,sameSite:'strict'
                });
            }
        })
    }
    return {exist:exist,response};
}
//it is a middleWare
async function verifyToken(req,res,next){
    console.log('in verify token');
    console.log(req.cookies)
    const {accessToken,refreshToken} =req.cookies;
    console.log(accessToken,refreshToken);
    if(!accessToken){   //if accessToken is not present , create it via renewToken()
        // if renewtoken present 
        const {exist,response}= renewToken(req,res,next)
        console.log('is renew token exist')
        console.log(exist);
        console.log(response);
        if(exist){//refreshToken returns a variable exist which is true if accessToken generated
            next(); 
        }else{
            return res.send(response);
        }
    }else{//if access token is present verify it
        jwt.verify(accessToken,process.env.ACCESS_TOKEN,async(err,decoded)=>{
            if(err){
                console.log('invalid access token');
                console.log(err);
                return res.send({status:400,msg:'invalid token'});
            }else{
                console.log('valid access token');
                console.log('decoded.email from verify token'+decoded.email);
                req.email=decoded.email;
                next();
            }
        })
    }
}

module.exports=verifyToken;