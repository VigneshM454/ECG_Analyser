import Component from '@glimmer/component';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';
import axios from 'axios';
import { inject as service } from '@ember/service';
export default class UploadFormComponent extends Component {
  @service ecgData;
  @service router;
  @tracked file = null;
  @tracked isNewPatient = false;

  constructor() {
    super(...arguments);
    console.log(this.args.data);
    this.isNewPatient = this.args.data ? false : true;
  }

  @tracked patient = {
    name: '',
    age: null,
    gender: 'Male',
  };

  @action updateName(event) {
    this.patient.name = event.target.value;
  }

  @action updateAge(event) {
    this.patient.age = event.target.value;
  }

  @action updateGender(gender) {
    this.patient.gender = gender;
  }

  @action fileSelected(event) {
    const file = event.target.files;
    if (file) {
      if (file.length > 4) {
        alert('You can only upload a maximum of 4 files.');
        event.target.value = ''; // reset input
        return;
      }
      this.file = file;
      console.log(this.file);
    }
  }

  @action getSpecific() {}

  @action async uploadFile() {
    event.preventDefault();
    console.log('hi im executed');
    console.log(this.patient);
    console.log(this.isNewPatient);
    console.log(this.file);

    // Prepare form data to send to the server
    const formData = new FormData();
    console.log(this.args.data);
    if(this.args.data){
      formData.append('patientId', this.args.data);
    }else{
      formData.append('name', this.patient.name);
      formData.append('age', this.patient.age);
      formData.append('gender', this.patient.gender);
    }
    for (var i = 0; i < this.file.length; i++) {
      formData.append('files', this.file[i]); // Append each file to the form data
    }

    const isvalid = this.validateFiles;
    if (!isvalid) return;
    // (this.file).forEach((fl) => {
    //     formData.append('files', fl); // Append each file to the form data
    // });

    axios
      .post(`${this.ecgData.baseUrl}/upload`, formData, { withCredentials: true })
      .then((res) => {
        console.log('Success:', res);
        alert("Ecg data uploaded successfully")
        this.router.transitionTo("home");
      })
      .catch((err) => {
        console.log(err.request);
        if (err.request.status == 401 || err.request.status == 403) {
          this.isAuthentic = false;
          this.router.transitionTo('index');
        }
      });
  }

  validateFiles() {
    // Extract extensions and base names
    if (this.file.length > 4) {
      alert('Only a maximum of 4 files can be uploaded at once');
      setEcgFiles([]);
      return false;
    }

    let baseNames = new Set();
    let extensions = new Set();

    for (let i = 0; i < this.file.length; i++) {
      const fileName = this.file[i].name;
      const parts = fileName.split('.');
      if (parts.length !== 2) {
        alert('Invalid file name format. Must be in the form name.ext');
        setEcgFiles([]);
        return false;
      }
      const [base, ext] = parts;
      baseNames.add(base);
      extensions.add(ext.toLowerCase());
    }

    // 1. Check .dat and .hea are present
    if (!(extensions.has('dat') && extensions.has('hea'))) {
      alert('Both .dat and .hea files are required for ECG analysis');
      setEcgFiles([]);
      return false;
    }

    // 2. Check all files have the same base name
    if (baseNames.size !== 1) {
      alert('All files must have the same base name (e.g., ecg.dat, ecg.hea)');
      setEcgFiles([]);
      return false;
    }
    return true;
  }
}

// if (err.request) {
//     console.log("No response received:", err.request);
//     alert("Server not responding. Check if the server is running.");
// } else {
//     console.log("Error setting up request:", err.message);
//     alert("Error sending request: " + err.message);
// }

// const formData = new FormData();
// for (let i = 0; i < this.file.length; i++) {
//   formData.append('file', this.file[i]); // ← name should be 'file'
// }
// // formData.append('files', this.file);

// try {
//   const response = await axios.post(
//     '${this.ecgData.baseUrl}/upload',
//     formData,
//     {
//       headers: {
//         'Content-Type': 'multipart/form-data',
//       },
//     },
//   );

//   console.log('File uploaded successfully:', response.data);
//   alert('File uploaded successfully!');
// } catch (error) {
//   console.error('Error uploading file:', error);
//   alert('Error uploading file.');
// }
