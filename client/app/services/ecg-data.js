import Service from '@ember/service';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';
import axios from 'axios';
import { inject as service } from '@ember/service';
export default class EcgDataService extends Service {
  @service router;

  @tracked user = {};
  @tracked patientList = [];
  @tracked recordList = [];
  @tracked isrecordFetched = false;
  @tracked isAuthentic;// = false;
  @tracked isAuthChecked = false;

  // baseUrl='https://ecg-analyser-1.onrender.com'//  http://localhost:3000

  baseUrl='http://localhost:3000'
  constructor(){
    super(...arguments);
    const authStatus = localStorage.getItem('authStatus');
    console.log(authStatus)
    console.log(typeof authStatus)
    console.log(authStatus ==='true');
    if(authStatus!=null){
      this.updateAuth('false')
    }

    this.updateAuth(authStatus);
  }

  @action updateAuth(status){
    console.log('inside updateAuth');
    console.log(status);
    localStorage.setItem('authStatus',status);
    if(status=='false'){
      console.log('inside if');
      this.router.transitionTo('index')
    }
    this.isAuthentic = status==='true';
    this.isAuthChecked=true;
  }

  @action getPatientList() {
    axios
      .get(`${this.baseUrl}/patients`, { withCredentials: true })
      .then((res) => {
        // alert(res.msg)
        console.log(res);
        console.log(res.data);
        this.patientList = [...res.data.patients];
      })
      .catch((err) => {
        if(err.status==401 ||  err.status==403){
          this.updateAuth('false')
        }
        alert(err.response.data.msg);
        console.log(err);
      });
  }

  async getRecordList(id) {
    try {
      const res = await axios.get(`${this.baseUrl}/patients/${id}`, {
        withCredentials: true,
      });

      console.log(res);
      console.log(res.data);

      return res.data; // ✅ return the data so caller gets it
    } catch (err) {
      alert(err.response?.data?.msg || 'Error fetching data');
      if(err.status==401 ||  err.status==403){
        this.updateAuth('false')
      }
      console.error(err);
      return null; // or throw if you want route to bubble error
    } finally {
      this.isrecordFetched = true;
    }
  }

  async getUniqueRecord(id) {
    try {
      const res = await axios.get(`${this.baseUrl}/records/${id}`, {
        withCredentials: true,
      });

      console.log(res);
      console.log(res.data);

      return res.data;
    } catch (err) {
      alert(err.response?.data?.msg || 'Error fetching data');
      if(err.status==401 ||  err.status==403){
        this.updateAuth('false')
      }
      console.error(err);
      return null; // or throw if you want route to bubble error
    }
  }

  @action async deletePatient(patientId){
    try {
      const res = await axios.delete(`${this.baseUrl}/patients/${patientId}`, {
        withCredentials: true,
      });

      console.log(res);
      console.log(res.data);
      alert("Patient Info delted Successfully")
      const { name, params } = this.router.currentRoute;
      this.router.transitionTo(name, ...Object.values(params));

    } catch (err) {
      if(err.status==401 ||  err.status==403){
        this.updateAuth('false')
      }
      alert(err.response?.data?.msg || 'Error deleteing Patient ');
      console.error(err);
    }
  }

  @action async deleteRecord(recordId){
    try {
      const res = await axios.delete(`${this.baseUrl}/records/${recordId}`, {
        withCredentials: true,
      });

      console.log(res);
      console.log(res.data);
      alert(`Ecg Record with id ${recordId} delted Successfully`)
      const { name, params } = this.router.currentRoute;
      this.router.transitionTo(name, ...Object.values(params));
    } catch (err) {
      if(err.status==401 ||  err.status==403){
        this.updateAuth('false')
      }
      alert(err.response?.data?.msg || 'Error deleteing Ecg Record ');
      console.error(err);
    }
  }

  @action async checkAuth() {
    try {
      const res = await axios.get(`${this.baseUrl}/isAuthentic`, {
        withCredentials: true,
      });
      this.updateAuth('true')
      // localStorage.setItem('authStatus')='true'
      // this.isAuthentic = true;
      console.log(res);
      this.user = res.data.userData;
      return true;
    } catch (err) {
      this.updateAuth('false')
      return false;
    } finally {
      this.isAuthChecked = true;
    }
  }

  @action async logout() {
    console.log('inside logout');
    axios
      .post(`${this.baseUrl}/logout`, {}, { withCredentials: true })
      .then((res) => {
        console.log(res);
      })
      .catch((err) => {
        console.log(err);
      })
      .finally(() => {
        this.updateAuth('false')
        this.router.transitionTo('index');
      });
  }
}

// @action getRecordList(id) {
//   axios
//     .get(`${this.baseUrl}patients/${id}`, { withCredentials: true })
//     .then((res) => {
//       // alert(res.msg)
//       console.log(res);
//       console.log(res.data);
//       // this.recordList = [...res.data.records];
//       return res.data
//     })
//     .catch((err) => {
//       alert(err.response.data.msg);
//       console.log(err);
//     })
//     .finally(() => (this.isrecordFetched = true));
// }
