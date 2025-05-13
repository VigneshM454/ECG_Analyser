const bcrypt=require('bcrypt');
const saltRounds=10;
//const crypto=require('crypto').randomBytes(32).toString('hex')
//console.log('crypto is ');
//console.log(crypto);
function hashPassword(password){
    const salt=bcrypt.genSaltSync(saltRounds);
    const hash= bcrypt.hashSync(password,salt);
    console.log(hash);
    return hash;
}

function comparePassword(plain,hash){
    const isValid=bcrypt.compareSync(plain,hash);
    console.log(isValid);
    return isValid;
}


module.exports={comparePassword,hashPassword}